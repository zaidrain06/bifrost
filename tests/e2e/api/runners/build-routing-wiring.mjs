#!/usr/bin/env node
// Generate the governance × model-catalog routing Postman collection.
//
// Each scenario stands up an isolated custom provider (backed by real OpenAI),
// adds gated keys, optionally creates a virtual key (VK) with provider/model/key
// restrictions, then drives inference and asserts the route taken. Two gates
// compose, evaluated in order:
//   1. governance  — VK provider allowlist, allowed/blacklisted models, key
//                    restriction. Rejects early (403/402/429).
//   2. catalog/key — per-key models/blacklist/aliases. Rejects at key selection
//                    with 400 "no keys found for provider: <p> and model: <m>".
//
// Verification:
//   * sync — on success assert extra_fields.routing_info {provider, model, key,
//     resolved_key_alias}; on rejection assert the error message (routing_info is
//     empty on errors raised before a route is chosen, and names the route on
//     errors after one is). Each routing request is sent once and judged once: before a
//     scenario's first inference request, a polling step waits for the model
//     catalog to list every provider's models (live discovery), and asserts nothing
//     about routing, so a route that only shows up on a retry fails.
//   * async — the stored log (polled), correlated by virtual_key_ids/providers:
//     status, selected_key_name, virtual_key presence. routing_engine_logs is
//     null for plain governance allow/deny, so it is NOT asserted here.
//
// VK key restriction is set via key_ids (["*"] = all keys); allow_all_keys/keys
// are ignored on input. Output is machine-generated — edit this script and re-run.
//
//   node build-routing-wiring.mjs [--out path.json]

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  url,
  events,
  request,
  item,
  folderPrerequest,
  pollPrerequest,
  pollTest,
  singleTest,
  catalogReadyAssertLines,
  mutationTest,
  cleanupTest,
  buildCollection,
  writeCollection,
  resolveOutPath,
} from "./lib/collection-builder.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT = join(HERE, "..", "collections", "bifrost-routing-wiring.postman_collection.json");

const NAME_PREFIX = "catwiring-rt-";

// Provider catalog. Key values use Bifrost's `env.<NAME>` resolution (credential
// read from the Bifrost process env at request time — no secret in the collection,
// nothing injected by the runner). base_url is omitted; the base type's default
// applies. Models are current, cheap, and asserted only as substrings/membership.
const PROVIDERS = {
  openai: { base: "openai", env: "OPENAI_API_KEY", model: "gpt-4o-mini", altModel: "gpt-4o" },
  anthropic: { base: "anthropic", env: "ANTHROPIC_API_KEY", model: "claude-haiku-4-5", altModel: "claude-sonnet-4-5" },
  gemini: { base: "gemini", env: "GEMINI_API_KEY", model: "gemini-2.5-flash", altModel: "gemini-2.5-flash-lite" },
  // Vertex is config-heavy and standard-only (no custom_provider_config). Its key
  // carries a vertex_key_config (project/region/credentials via env.) instead of a
  // value. region "global" serves the cross-region Claude models.
  vertex: {
    base: "vertex",
    standardOnly: true,
    model: "claude-sonnet-4-5",
    keyConfig: {
      vertex_key_config: {
        project_id: "env.VERTEX_PROJECT_ID",
        project_number: "env.VERTEX_PROJECT_NUMBER",
        region: "global",
        auth_credentials: "env.VERTEX_CREDENTIALS",
      },
    },
  },
  // Azure: standard-only; key carries the api key (value) plus azure_key_config.
  // Deployments are named after the model, so gpt-4o-mini routes directly.
  azure: {
    base: "azure",
    standardOnly: true,
    model: "gpt-4o-mini",
    keyConfig: { value: "env.AZURE_API_KEY", azure_key_config: { endpoint: "env.AZURE_ENDPOINT" } },
  },
  // Bedrock: standard-only; bedrock_key_config (IAM creds + region). Claude needs
  // the cross-region inference-profile id ("us.anthropic.…").
  bedrock: {
    base: "bedrock",
    standardOnly: true,
    // Routed via a key alias so Bedrock exposes the common Claude name; the alias
    // resolves to the cross-region inference-profile id Bedrock requires.
    model: "claude-sonnet-4-5",
    aliasFor: { "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0" },
    keyConfig: {
      bedrock_key_config: { access_key: "env.AWS_ACCESS_KEY_ID", secret_key: "env.AWS_SECRET_ACCESS_KEY", region: "env.AWS_REGION" },
    },
  },
};
const DEFAULT_PROVIDER = "openai";

// Back-compat aliases for the openai-based scenarios.
const MODEL_A = PROVIDERS.openai.altModel; // gpt-4o
const MODEL_B = PROVIDERS.openai.model; // gpt-4o-mini

// --------------------------------------------------------------------------- //
// Spec helpers
// --------------------------------------------------------------------------- //

function key({ id, models = [], blacklisted = [], enabled = true, aliases = {}, weight, badKey = false } = {}) {
  return { id, models, blacklisted, enabled, aliases, weight, badKey };
}

// Key names must be unique across all providers, so namespace by scenario + key
// id + run id. Assertions reference a key by its id and recompute this name.
// prov is the PROVIDERS entry: a value=env.<NAME> key for simple providers, or a
// provider-specific key config (e.g. vertex_key_config) when prov.keyConfig is set.
function keyBody(k, name, prov) {
  const out = { id: keyId(k.id), name, models: [...k.models], enabled: k.enabled };
  if (k.badKey) {
    // A deliberately invalid credential so the provider attempt fails upstream
    // (e.g. a 401), exercising the fallback path. The provider itself is real
    // and capable of the model — only the credential is broken.
    out.value = "sk-deadbeef-invalid-000";
  } else if (prov.keyConfig) Object.assign(out, prov.keyConfig);
  else out.value = `env.${prov.env}`;
  if (k.blacklisted.length) out.blacklisted_models = [...k.blacklisted];
  // prov.aliasFor lets a provider expose a common model name (e.g. Bedrock maps
  // claude-sonnet-4-5 → its inference-profile id). Scenario aliases win on conflict.
  const aliases = { ...(prov.aliasFor || {}), ...k.aliases };
  if (Object.keys(aliases).length) out.aliases = aliases;
  if (k.weight != null) out.weight = k.weight;
  return out;
}

const keyNameSeg = (sid, kid) => `catwiring-rt-${sid}-${kid}-{{run_id}}`;
const jsKeyName = (sid, kid) => `'catwiring-rt-${sid}-${kid}-' + pm.variables.get('run_id')`;
// JS expression for a provider's run-scoped name (ref "self" = primary provider).
const jsSegFor = (sid, ref) =>
  !ref || ref === "self"
    ? `'${NAME_PREFIX}${sid}-' + pm.variables.get('run_id')`
    : `'${NAME_PREFIX}${sid}-${ref}-' + pm.variables.get('run_id')`;

// A VK provider config. provider_ref 'self' targets this scenario's provider.
function vkProvider({ providerRef = "self", keyIds = ["*"], allowedModels = ["*"], blacklistedModels = [], weight = 1 } = {}) {
  return { providerRef, keyIds, allowedModels, blacklistedModels, weight };
}

const keyId = (kid) => `${kid}-{{run_id}}`;
const providerSeg = (sid) => NAME_PREFIX + sid + "-{{run_id}}";
const jsProviderName = (sid) => `'${NAME_PREFIX}${sid}-' + pm.variables.get('run_id')`;

// --------------------------------------------------------------------------- //
// Assertion line builders
// --------------------------------------------------------------------------- //

// Header checks shared by success and rejection: the routing-info provider header names a provider,
// and no response header carries a given prefix (a fallback's provider headers leaking through).
function routeHeaderLines(step, jsNameOf) {
  const lines = [];
  if (step.expectHeaderProviderRef != null) {
    lines.push(`var headerProvider = ${jsNameOf(step.expectHeaderProviderRef)};`);
    lines.push("if (pm.response.headers.get('x-bifrost-routing-info-provider') !== headerProvider) { throw new Error('x-bifrost-routing-info-provider=' + pm.response.headers.get('x-bifrost-routing-info-provider') + ' expected ' + headerProvider); }");
  }
  if (step.expectNoHeaderPrefix) {
    lines.push(`var bannedPrefix = ${JSON.stringify(step.expectNoHeaderPrefix.toLowerCase())};`);
    lines.push("pm.response.headers.each(function (h) { if (String(h.key).toLowerCase().indexOf(bannedPrefix) === 0) { throw new Error('response carries header ' + h.key + ': ' + h.value); } });");
  }
  return lines;
}

function routeAssertLines(sid, step, jsNameOf) {
  if (step.expectStatus === 200 && step.endpoint === "anthropic") {
    // The /anthropic route answers in the Messages shape; routing info rides on the headers.
    const lines = [
      "if (pm.response.code !== 200) { throw new Error('route status ' + pm.response.code + ' body ' + pm.response.text()); }",
      "var body = pm.response.json();",
      "if (body.type !== 'message' || !Array.isArray(body.content) || body.content.length === 0) { throw new Error('not a Messages-shaped answer: ' + pm.response.text()); }",
    ];
    if (step.expectProviderOneOf) {
      lines.push(`var allowedProviders = [${step.expectProviderOneOf.map((r) => jsNameOf(r)).join(", ")}];`);
      lines.push("var served = pm.response.headers.get('x-bifrost-routing-info-provider');");
      lines.push("if (allowedProviders.indexOf(served) < 0) { throw new Error('x-bifrost-routing-info-provider=' + served + ' not in ' + JSON.stringify(allowedProviders)); }");
    }
    return [...lines, ...routeHeaderLines(step, jsNameOf)];
  }
  if (step.expectStatus === 200) {
    const lines = [
      "if (pm.response.code !== 200) { throw new Error('route status ' + pm.response.code + ' body ' + pm.response.text()); }",
      "var body = pm.response.json();",
      "if (!body.choices || body.choices.length === 0) { throw new Error('no choices'); }",
      "var ri = (body.extra_fields || {}).routing_info || {};",
      ...routeHeaderLines(step, jsNameOf),
    ];
    if (step.expectProviderOneOf) {
      lines.push(`var allowedProviders = [${step.expectProviderOneOf.map((r) => jsNameOf(r)).join(", ")}];`);
      lines.push("if (allowedProviders.indexOf(ri.provider) < 0) { throw new Error('routing_info.provider=' + ri.provider + ' not in ' + JSON.stringify(allowedProviders)); }");
    } else {
      lines.push(`var providerName = ${jsNameOf(step.routeRef)};`);
      lines.push("if (ri.provider !== providerName) { throw new Error('routing_info.provider=' + ri.provider + ' expected ' + providerName); }");
    }
    if (step.expectKeyId != null) {
      lines.push(`var expectedKey = ${jsKeyName(sid, step.expectKeyId)};`);
      lines.push("if (ri.key !== expectedKey) { throw new Error('routing_info.key=' + ri.key + ' expected ' + expectedKey); }");
    }
    if (step.expectResolvedModelId != null) {
      lines.push(
        `if (!ri.resolved_key_alias || ri.resolved_key_alias.model_id !== ${JSON.stringify(step.expectResolvedModelId)}) { throw new Error('resolved_key_alias=' + JSON.stringify(ri.resolved_key_alias)); }`
      );
    }
    if (step.expectIsFallback != null) {
      lines.push(`if (Boolean(ri.is_fallback) !== ${JSON.stringify(step.expectIsFallback)}) { throw new Error('is_fallback=' + ri.is_fallback); }`);
    }
    if (step.expectPrimaryProviderRef != null) {
      lines.push(`var expectedPrimary = ${jsNameOf(step.expectPrimaryProviderRef)};`);
      lines.push("if (ri.primary_provider !== expectedPrimary) { throw new Error('primary_provider=' + ri.primary_provider + ' expected ' + expectedPrimary); }");
    }
    if (step.expectPrimaryModel != null) {
      lines.push(`if (ri.primary_model !== ${JSON.stringify(step.expectPrimaryModel)}) { throw new Error('primary_model=' + ri.primary_model); }`);
    }
    return lines;
  }
  const lines = [
    `if (pm.response.code !== ${step.expectStatus}) { throw new Error('expected status ${step.expectStatus} got ' + pm.response.code + ' body ' + pm.response.text()); }`,
    "var body = pm.response.json();",
    "var msg = (body.error && (body.error.message || body.error)) || body.message || '';",
  ];
  if (step.expectErrorSubstr) {
    // Run variables such as {{run_id}} are not substituted inside script source, so resolve them here.
    lines.push(`var expectedSubstr = pm.variables.replaceIn(${JSON.stringify(step.expectErrorSubstr)});`);
    lines.push("if (String(msg).indexOf(expectedSubstr) < 0) { throw new Error('error message=' + msg + ' expected to contain ' + expectedSubstr); }");
  }
  if (step.expectErrorType) {
    lines.push(`if (body.type !== ${JSON.stringify(step.expectErrorType)}) { throw new Error('error type=' + body.type + ' expected ${step.expectErrorType}'); }`);
  }
  return [...lines, ...routeHeaderLines(step, jsNameOf)];
}

function logAssertLines(sid, step, jsNameOf) {
  const lines = [
    `var providerName = ${jsNameOf(step.providerRef)};`,
    "if (pm.response.code !== 200) { throw new Error('logs status ' + pm.response.code); }",
    "var logs = (pm.response.json() || {}).logs || [];",
    `var model = ${JSON.stringify(step.model)};`,
    "var row = null;",
    "for (var i = 0; i < logs.length; i++) {",
    "  if (logs[i].provider === providerName && logs[i].model === model) { row = logs[i]; break; }",
    "}",
    "if (!row) { throw new Error('no log row for ' + providerName + '/' + model + ' in ' + logs.length + ' rows'); }",
    `if (row.status !== ${JSON.stringify(step.expectStatus)}) { throw new Error('log status=' + row.status); }`,
  ];
  if (step.expectSelectedKeyId != null) {
    lines.push(`var expectedKey = ${jsKeyName(sid, step.expectSelectedKeyId)};`);
    lines.push("if (row.selected_key_name !== expectedKey) { throw new Error('selected_key_name=' + row.selected_key_name + ' expected ' + expectedKey); }");
  }
  if (step.expectVkPresent) {
    lines.push("if (!row.virtual_key_id) { throw new Error('log row missing virtual_key_id'); }");
  }
  if (step.expectFallbackIndex != null) {
    lines.push(`if (row.fallback_index !== ${JSON.stringify(step.expectFallbackIndex)}) { throw new Error('fallback_index=' + row.fallback_index + ' expected ${step.expectFallbackIndex}'); }`);
  }
  return lines;
}

// One sample of a distribution probe: assert 200 and record which key served into
// a scenario-scoped set. `reset` clears the set on the first sample.
function distSampleTest(testname, sid, cleanupName, reset) {
  return [
    `var cleanupReq = ${JSON.stringify(cleanupName)};`,
    ...(reset ? [`pm.collectionVariables.set('dist_${sid}', '');`] : []),
    "if (pm.response.code === 200) {",
    "  var k = ((pm.response.json().extra_fields || {}).routing_info || {}).key || '';",
    `  var cur = pm.collectionVariables.get('dist_${sid}') || '';`,
    "  var set = cur ? cur.split(',') : [];",
    `  if (k && set.indexOf(k) < 0) { set.push(k); pm.collectionVariables.set('dist_${sid}', set.join(',')); }`,
    "}",
    `pm.test(${JSON.stringify(testname)}, function () {`,
    "  pm.expect(pm.response.code, 'status ' + pm.response.code + ' body ' + pm.response.text()).to.equal(200);",
    "});",
    "if (pm.response.code !== 200) { pm.execution.setNextRequest(cleanupReq); }",
  ];
}

// One sample of a provider-distribution probe: assert 200 and record which
// provider served into a scenario-scoped set.
function distSampleProviderTest(testname, sid, cleanupName, reset) {
  return [
    `var cleanupReq = ${JSON.stringify(cleanupName)};`,
    ...(reset ? [`pm.collectionVariables.set('pdist_${sid}', '');`] : []),
    "if (pm.response.code === 200) {",
    "  var p = ((pm.response.json().extra_fields || {}).routing_info || {}).provider || '';",
    `  var cur = pm.collectionVariables.get('pdist_${sid}') || '';`,
    "  var set = cur ? cur.split(',') : [];",
    `  if (p && set.indexOf(p) < 0) { set.push(p); pm.collectionVariables.set('pdist_${sid}', set.join(',')); }`,
    "}",
    `pm.test(${JSON.stringify(testname)}, function () {`,
    "  pm.expect(pm.response.code, 'status ' + pm.response.code + ' body ' + pm.response.text()).to.equal(200);",
    "});",
    "if (pm.response.code !== 200) { pm.execution.setNextRequest(cleanupReq); }",
  ];
}

// Assert every expected key was observed across the distribution samples.
// Assert the observed key set against the step's expectations:
//   expectKeyIds  — each of these keys must have served at least once
//   expectOnly    — every observed key must be one of these
//   expectNever   — none of these keys may have served
function distAssertLines(sid, step) {
  const arr = (ids) => "[" + (ids || []).map((kid) => jsKeyName(sid, kid)).join(", ") + "]";
  const lines = [`var seen = (pm.collectionVariables.get('dist_${sid}') || '').split(',').filter(Boolean);`];
  if (step.expectKeyIds) {
    lines.push(`${"var"} mustServe = ${arr(step.expectKeyIds)};`);
    lines.push("mustServe.forEach(function (e) { if (seen.indexOf(e) < 0) throw new Error('key ' + e + ' never served; observed ' + JSON.stringify(seen)); });");
  }
  if (step.expectOnly) {
    lines.push(`var only = ${arr(step.expectOnly)};`);
    lines.push("if (seen.length === 0) throw new Error('no key observed across samples');");
    lines.push("seen.forEach(function (e) { if (only.indexOf(e) < 0) throw new Error('key ' + e + ' served but not in expectOnly ' + JSON.stringify(only)); });");
  }
  if (step.expectNever) {
    lines.push(`var never = ${arr(step.expectNever)};`);
    lines.push("seen.forEach(function (e) { if (never.indexOf(e) >= 0) throw new Error('key ' + e + ' served but was expectNever; observed ' + JSON.stringify(seen)); });");
  }
  return lines;
}

// Capture the created VK's value+id into scenario-scoped collection vars.
function captureVkTest(testname, sid, cleanupName) {
  return [
    `var cleanupReq = ${JSON.stringify(cleanupName)};`,
    "var ok = pm.response.code === 200 || pm.response.code === 201;",
    `pm.test(${JSON.stringify(testname)}, function () {`,
    "  pm.expect([200, 201], 'status ' + pm.response.code + ' body ' + pm.response.text()).to.include(pm.response.code);",
    "});",
    "if (ok) {",
    "  var vk = (pm.response.json() || {}).virtual_key || {};",
    `  pm.collectionVariables.set('vkval_${sid}', vk.value || '');`,
    `  pm.collectionVariables.set('vkid_${sid}', vk.id || '');`,
    // Each provider config's id by provider name, so an update can edit a config in place.
    "  var pcIds = {};",
    "  (vk.provider_configs || []).forEach(function (pc) { pcIds[pc.provider] = pc.id; });",
    `  pm.collectionVariables.set('vkpc_${sid}', JSON.stringify(pcIds));`,
    "} else { pm.execution.setNextRequest(cleanupReq); }",
  ];
}

// The provider configs of a VK update, built at request time so each carries the id of the config it
// edits (captured at create; an undefined id is left out). Each config is an object literal inside
// the one JSON.stringify that builds the body, so its field order is fixed. weight undefined omits the
// field; weight null sends JSON null.
function vkUpdatePrerequest(sid, configs, jsNameOf) {
  const literals = configs.map((pc) => {
    const name = jsNameOf(pc.providerRef);
    const weight = pc.weight === undefined ? "" : `, weight: ${JSON.stringify(pc.weight)}`;
    return `{ provider: ${name}, allowed_models: ${JSON.stringify(pc.allowedModels || ["*"])}, key_ids: ["*"], id: pcIds[${name}] != null ? pcIds[${name}] : undefined${weight} }`;
  });
  return [
    `var pcIds = JSON.parse(pm.collectionVariables.get('vkpc_${sid}') || '{}');`,
    `pm.request.body.update(JSON.stringify({ provider_configs: [${literals.join(", ")}] }));`,
  ];
}

// Assert a VK read back from GET /api/governance/virtual-keys/{id} stores weight on provider's
// config (null: no weight stored).
function vkWeightAssertLines(step, jsNameOf) {
  return [
    "if (pm.response.code !== 200) { throw new Error('get vk status ' + pm.response.code + ' body ' + pm.response.text()); }",
    "var vk = (pm.response.json() || {}).virtual_key || {};",
    `var name = ${jsNameOf(step.providerRef)};`,
    "var pc = (vk.provider_configs || []).filter(function (c) { return c.provider === name; })[0];",
    "if (!pc) { throw new Error('no provider config for ' + name + ' in ' + JSON.stringify(vk.provider_configs)); }",
    `var want = ${JSON.stringify(step.expectWeight)};`,
    "var got = pc.weight === undefined ? null : pc.weight;",
    "if (got !== want) { throw new Error('stored weight=' + JSON.stringify(pc.weight) + ' expected ' + JSON.stringify(want)); }",
  ];
}

// --------------------------------------------------------------------------- //
// Step expansion
// --------------------------------------------------------------------------- //

function expandScenario(sc) {
  const sid = sc.id;
  const seg = providerSeg(sid);
  const cleanupProvider = `cleanup: delete provider [${sid}]`;
  const cleanupVk = `cleanup: delete vk [${sid}]`;
  // A failing step jumps to the FIRST cleanup item so the whole teardown runs in
  // order. When the scenario has a VK, that first item is the VK delete (the
  // provider delete follows it); jumping straight to the provider delete would
  // skip the VK and leak it.
  const hasVkScenario = sc.steps.some((s) => s.type === "createVK");
  // A scenario may create several model routers, each under its own ref ("r1", "r2"). A router's id
  // and name are the same run-scoped string, and every router created is deleted in the teardown,
  // before the virtual key and the providers its targets name.
  const routerRefs = [...new Set(sc.steps.filter((s) => s.type === "createRouter").map((s) => s.ref || "r1"))];
  const routerIdSeg = (ref) => `catwiring-rtr-${sid}-${ref || "r1"}-{{run_id}}`;
  const cleanupRouter = (ref) => `cleanup: delete model router ${ref} [${sid}]`;
  const cleanupTarget = routerRefs.length ? cleanupRouter(routerRefs[0]) : hasVkScenario ? cleanupVk : cleanupProvider;
  // A scenario may stand up more than one provider (e.g. governance LB across
  // providers). ref "self" is the scenario's primary provider; any other ref
  // gets its own run-scoped name. All created providers are torn down.
  const segFor = (ref) => (!ref || ref === "self" ? seg : `${NAME_PREFIX}${sid}-${ref}-{{run_id}}`);
  // Resolve each provider ref (kind-aware). A "custom" provider is run-namespaced
  // (parallel-safe); a "standard" provider uses the fixed base-type name (e.g.
  // "openai") — a GLOBAL singleton, so standard-provider scenarios are NOT
  // run-id-isolated: they assume a clean instance and must not run in parallel
  // shards touching the same standard provider.
  const providerByRef = {};
  for (const s of sc.steps) {
    if (s.type !== "addProvider") continue;
    const ref = s.ref || "self";
    const pt = s.providerType || DEFAULT_PROVIDER;
    const std = s.providerKind === "standard" || !!PROVIDERS[pt].standardOnly;
    providerByRef[ref] = {
      pt,
      std,
      env: PROVIDERS[pt].env,
      base: PROVIDERS[pt].base,
      name: std ? PROVIDERS[pt].base : segFor(ref),
      jsName: std ? JSON.stringify(PROVIDERS[pt].base) : jsSegFor(sid, ref),
    };
  }
  const providerRefs = Object.keys(providerByRef).length ? Object.keys(providerByRef) : ["self"];
  const nameOf = (ref) => (providerByRef[ref || "self"] || {}).name || segFor(ref);
  const jsNameOf = (ref) => (providerByRef[ref || "self"] || {}).jsName || jsSegFor(sid, ref);
  const items = [];
  let counter = 0;
  let ordinal = 0;
  let hasVk = false;
  const nextId = (tag) => `rt-${sid}-${String(++counter).padStart(2, "0")}-${tag}`;
  const uniq = (label) => `${String(++ordinal).padStart(2, "0")}. ${label} [${sid}]`;

  // Providers added since the last catalog wait. Before the next inference request, one polling
  // step per provider waits for the catalog to list its models: the list is non-empty (live
  // discovery landed) and holds every model its enabled keys name explicitly. Only that wait
  // polls; the routing requests after it are sent once.
  let catalogPending = [];
  const awaitCatalog = () => {
    for (const { ref, keys } of catalogPending) {
      const enabled = keys.filter((k) => k.enabled);
      if (enabled.length === 0) continue; // nothing to discover: no key can serve
      const models = [];
      for (const k of enabled) {
        for (const m of k.models) {
          if (m !== "*" && !k.blacklisted.includes(m) && !models.includes(m)) models.push(m);
        }
      }
      const name = uniq(`wait for the catalog to list ${ref === "self" ? "the provider" : ref}'s models`);
      const query = [{ key: "provider", value: nameOf(ref) }, { key: "limit", value: "1000" }];
      items.push(item(nextId("await-catalog"), name, request("GET", url(["api", "models"], query), null),
        events(pollPrerequest(0), pollTest(name, catalogReadyAssertLines(jsNameOf(ref), models), cleanupTarget))));
    }
    catalogPending = [];
  };
  const inferenceSteps = new Set(["route", "keyDistribution", "providerDistribution"]);

  for (const step of sc.steps) {
    if (inferenceSteps.has(step.type) && catalogPending.length) awaitCatalog();
    switch (step.type) {
      case "addProvider": {
        const pseg = nameOf(step.ref);
        const prov = PROVIDERS[step.providerType || DEFAULT_PROVIDER];
        const plabel = step.ref && step.ref !== "self" ? ` (${step.ref})` : "";
        // Standard providers (fixed base-type name) carry no custom_provider_config —
        // the handler rejects a custom config on a standard provider name.
        const isStandard = step.providerKind === "standard" || !!prov.standardOnly;
        const body = { provider: pseg };
        if (!isStandard) {
          body.custom_provider_config = { base_provider_type: prov.base, is_key_less: false };
        }
        const name = uniq("add provider" + plabel);
        items.push(item(nextId("add-provider"), name, request("POST", url(["api", "providers"]), body),
          events(null, mutationTest(name, [200, 201], cleanupTarget))));
        for (const k of step.keys || []) {
          const kname = uniq("add key " + k.id + plabel);
          items.push(item(nextId("add-key"), kname,
            request("POST", url(["api", "providers", pseg, "keys"]), keyBody(k, keyNameSeg(sid, k.id), prov)),
            events(null, mutationTest(kname, [200, 201], cleanupTarget))));
        }
        catalogPending.push({ ref: step.ref || "self", keys: step.keys || [] });
        break;
      }
      case "createVK": {
        hasVk = true;
        const pcs = (step.providerConfigs || []).map((pc) => {
          const cfg = {
            provider: nameOf(pc.providerRef),
            // "*" stays literal; a logical key id (e.g. "k1") is namespaced to the
            // run-scoped id the key was created with.
            key_ids: pc.keyIds.map((kid) => (kid === "*" ? "*" : keyId(kid))),
            allowed_models: [...pc.allowedModels],
            blacklisted_models: [...pc.blacklistedModels],
          };
          // weight:null → omit the field entirely (a VK with no weighted configs
          // is allow-list-only: governance gates providers but skips load balancing).
          if (pc.weight !== null) cfg.weight = pc.weight ?? 1;
          return cfg;
        });
        const body = { name: `catwiring-rtvk-${sid}-{{run_id}}`, is_active: true, provider_configs: pcs };
        const name = uniq("create vk");
        items.push(item(nextId("create-vk"), name, request("POST", url(["api", "governance", "virtual-keys"]), body),
          events(null, captureVkTest(name, sid, cleanupTarget))));
        break;
      }
      case "route": {
        const name = uniq(step.label);
        const headers = step.useVk === false ? [] : [{ key: "x-bf-vk", value: `{{vkval_${sid}}}` }];
        // pinKey pins the request to one of the scenario's keys (x-bf-api-key-id), for a check that
        // needs a particular key rather than whichever the weighted pick lands on.
        if (step.pinKey) {
          headers.push({ key: "x-bf-api-key-id", value: keyId(step.pinKey) });
        }
        const body = {
          // bareModel routes without a provider prefix so an upstream routing
          // layer (governance LB) resolves the provider. routeRef targets a
          // specific provider in a multi-provider scenario (default: self).
          model: step.bareModel ? step.model : `${nameOf(step.routeRef)}/${step.model}`,
          messages: [{ role: "user", content: "Reply with the single word: ok." }],
          max_tokens: 5,
        };
        // Request-level fallbacks. The /v1 OpenAI-compatible endpoint takes the
        // string "provider/model" form (not the {provider,model} object form).
        if (step.fallbacks) {
          body.fallbacks = step.fallbacks.map((f) => `${nameOf(f.providerRef)}/${f.model}`);
        }
        // endpoint: "v1" (default) is /v1/chat/completions, "openai" the OpenAI drop-in route and
        // "anthropic" the Anthropic Messages route (same body: model, messages, max_tokens).
        const path = { v1: ["v1", "chat", "completions"], openai: ["openai", "v1", "chat", "completions"], anthropic: ["anthropic", "v1", "messages"] }[step.endpoint || "v1"];
        // requestId sends a known x-request-id, kept as reqid_<sid>, so later steps can read this
        // request's own log row (its id) and its fallback attempts' rows (their parent_request_id).
        const pre = step.requestId ? [
          `var rid = 'rw-${sid}-${String(ordinal).padStart(2, "0")}-' + pm.variables.get('run_id');`,
          `pm.collectionVariables.set('reqid_${sid}', rid);`,
          "pm.request.headers.upsert({ key: 'x-request-id', value: rid });",
        ] : null;
        items.push(item(nextId("route"), name, request("POST", url(path), body, headers),
          events(pre, singleTest(name, routeAssertLines(sid, step, jsNameOf), cleanupTarget))));
        break;
      }
      case "updateKey": {
        // A key PUT is a full-field replace, so the whole intended key is sent.
        const ref = step.ref || "self";
        const name = uniq(step.label);
        const prov = PROVIDERS[providerByRef[ref].pt];
        items.push(item(nextId("update-key"), name,
          request("PUT", url(["api", "providers", nameOf(ref), "keys", keyId(step.key.id)]), keyBody(step.key, keyNameSeg(sid, step.key.id), prov)),
          events(null, mutationTest(name, [200], cleanupTarget))));
        break;
      }
      case "aliasPricing": {
        // The catalog's listing of an alias: /api/models/details prices the alias row from one key's
        // target. The price is kept as aliasprice_<sid>_<as>; with higherThan it must also be above a
        // price kept earlier.
        const name = uniq(step.label);
        const query = [{ key: "provider", value: nameOf(step.ref) }, { key: "limit", value: "1000" }];
        const lines = [
          "if (pm.response.code !== 200) { throw new Error('model details status ' + pm.response.code); }",
          `var alias = pm.variables.replaceIn(${JSON.stringify(step.alias)});`,
          `var providerName = ${jsNameOf(step.ref)};`,
          "var row = ((pm.response.json() || {}).models || []).filter(function (m) { return m.provider === providerName && m.name === alias; })[0];",
          "if (!row) { throw new Error('no details row for ' + alias); }",
          "if (typeof row.input_cost_per_token !== 'number') { throw new Error('the alias row carries no price: ' + JSON.stringify(row)); }",
          `pm.collectionVariables.set('aliasprice_${sid}_${step.as}', String(row.input_cost_per_token));`,
          ...(step.higherThan ? [
            `var earlier = parseFloat(pm.collectionVariables.get('aliasprice_${sid}_${step.higherThan}'));`,
            `if (!(row.input_cost_per_token > earlier)) { throw new Error('alias priced ' + row.input_cost_per_token + ', expected above the ${step.higherThan} price ' + earlier); }`,
          ] : []),
        ];
        items.push(item(nextId("alias-pricing"), name, request("GET", url(["api", "models", "details"], query), null),
          events(pollPrerequest(0), pollTest(name, lines, cleanupTarget))));
        break;
      }
      case "assertFallbackRow": {
        // The log row of one fallback attempt of the last requestId route: attempts after the
        // primary are logged under it as their parent_request_id.
        const name = uniq(step.label);
        const query = [{ key: "parent_request_id", value: `{{reqid_${sid}}}` }, { key: "limit", value: "10" }];
        const lines = [
          "if (pm.response.code !== 200) { throw new Error('logs status ' + pm.response.code); }",
          "var logs = (pm.response.json() || {}).logs || [];",
          `var providerName = ${jsNameOf(step.providerRef)};`,
          `var row = logs.filter(function (l) { return l.provider === providerName && l.fallback_index === ${step.fallbackIndex}; })[0];`,
          `if (!row) { throw new Error('no fallback attempt on ' + providerName + ' at index ${step.fallbackIndex} among ' + JSON.stringify(logs.map(function (l) { return [l.provider, l.fallback_index, l.status]; }))); }`,
          `if (row.status !== ${JSON.stringify(step.expectStatus)}) { throw new Error('fallback attempt status=' + row.status); }`,
          ...(step.model ? [`if (row.model !== ${JSON.stringify(step.model)}) { throw new Error('fallback attempt model=' + row.model); }`] : []),
        ];
        items.push(item(nextId("assert-fallback-row"), name, request("GET", url(["api", "logs"], query), null),
          events(pollPrerequest(step.waitSeconds), pollTest(name, lines, cleanupTarget))));
        break;
      }
      case "createRouter": {
        // A model router in the global scope, on and taking the requests cel holds for. Its condition
        // should match only this run's own model, so it cannot take another scenario's requests. It
        // is a conditional router unless the step names a mode; a classifier router's settings come
        // from step.classifier, a function of the provider names. Targets and fallbacks name
        // providers by ref, a classifier router's targets also name a tier, and a target with no
        // providerRef and no model keeps the request's own. A conditional router's fallbacks are the
        // step's own, one list for whichever target is chosen; a classifier router's are on its
        // targets. Targets are saved on their own endpoint, with the router's fallbacks, as the
        // dashboard saves them.
        const ref = step.ref || "r1";
        const name = uniq(step.label);
        const routing = { enabled: true, scope: "global", cel_expression: step.cel, priority: step.priority ?? 0 };
        if (step.chain) routing.chain = true;
        if (step.emptyTarget) routing.empty_target = step.emptyTarget;
        const body = {
          id: routerIdSeg(ref),
          name: routerIdSeg(ref),
          mode: step.mode || "conditional",
          ...(step.classifier ? step.classifier(nameOf) : {}),
          routing,
        };
        items.push(item(nextId("create-router"), name, request("POST", url(["api", "routing", "model-routers"]), body),
          events(null, mutationTest(name, [200, 201], cleanupTarget))));
        if (step.targets) {
          const targets = step.targets.map((tg) => ({
            ...(tg.tier ? { tier: tg.tier } : {}),
            ...(tg.providerRef ? { provider: nameOf(tg.providerRef) } : {}),
            ...(tg.model ? { model: tg.model } : {}),
            weight: tg.weight ?? 1,
            ...(tg.fallbacks ? { fallbacks: tg.fallbacks.map((f) => `${nameOf(f.providerRef)}/${f.model}`) } : {}),
          }));
          const fallbacks = step.fallbacks ? { fallbacks: step.fallbacks.map((f) => `${nameOf(f.providerRef)}/${f.model}`) } : {};
          const tname = uniq(`set the targets of router ${ref}`);
          items.push(item(nextId("router-targets"), tname,
            request("PUT", url(["api", "routing", "model-routers", routerIdSeg(ref), "targets"]), { targets, ...fallbacks }),
            events(null, mutationTest(tname, [200], cleanupTarget))));
        }
        break;
      }
      case "routerApi": {
        // One management call on the model router API, sent once and judged once: its status and,
        // for a refusal, a part of its message. With ref it is a call on that router (path names what
        // follows the id, such as ["routing"] or ["targets"]); without ref it is a call on the
        // collection, such as a create the API is expected to refuse. body may be a function of the
        // provider names and the router ids, for a body that names them.
        const name = uniq(step.label);
        const path = ["api", "routing", "model-routers", ...(step.ref ? [routerIdSeg(step.ref)] : []), ...(step.path || [])];
        const body = typeof step.body === "function" ? step.body({ nameOf, routerId: routerIdSeg }) : (step.body ?? null);
        const lines = [
          `if (pm.response.code !== ${step.expectStatus}) { throw new Error('expected status ${step.expectStatus} got ' + pm.response.code + ' body ' + pm.response.text()); }`,
          ...(step.expectErrorSubstr ? [
            "var body = pm.response.json() || {};",
            "var msg = (body.error && (body.error.message || body.error)) || body.message || '';",
            `var expectedSubstr = pm.variables.replaceIn(${JSON.stringify(step.expectErrorSubstr)});`,
            "if (String(msg).indexOf(expectedSubstr) < 0) { throw new Error('error message=' + msg + ' expected to contain ' + expectedSubstr); }",
          ] : []),
        ];
        items.push(item(nextId("router-api"), name, request(step.method, url(path), body),
          events(null, singleTest(name, lines, cleanupTarget))));
        break;
      }
      case "deleteProvider": {
        const name = uniq(step.label);
        items.push(item(nextId("delete-provider"), name, request("DELETE", url(["api", "providers", nameOf(step.ref)]), null),
          events(null, mutationTest(name, [200, 204], cleanupTarget))));
        // A deleted provider has no models left to wait for.
        catalogPending = catalogPending.filter((p) => p.ref !== step.ref);
        break;
      }
      case "updateVK": {
        const name = uniq(step.label);
        const req = request("PUT", url(["api", "governance", "virtual-keys", `{{vkid_${sid}}}`]), {});
        const status = step.expectStatus;
        const exec = [
          `var cleanupReq = ${JSON.stringify(cleanupTarget)};`,
          `pm.test(${JSON.stringify(name)}, function () {`,
          `  pm.expect(pm.response.code, 'status ' + pm.response.code + ' body ' + pm.response.text()).to.equal(${status});`,
          ...(step.expectErrorSubstr ? [`  pm.expect(pm.response.text()).to.include(${JSON.stringify(step.expectErrorSubstr)});`] : []),
          "});",
          `if (pm.response.code !== ${status}) { pm.execution.setNextRequest(cleanupReq); }`,
        ];
        items.push(item(nextId("update-vk"), name, req, events(vkUpdatePrerequest(sid, step.providerConfigs, jsNameOf), exec)));
        break;
      }
      case "assertVKWeight": {
        const name = uniq(step.label);
        items.push(item(nextId("get-vk"), name, request("GET", url(["api", "governance", "virtual-keys", `{{vkid_${sid}}}`]), null),
          events(null, singleTest(name, vkWeightAssertLines(step, jsNameOf), cleanupTarget))));
        break;
      }
      case "assertLog": {
        const name = uniq(step.label);
        // byVk:false reads the rows of providerRef (default: the scenario's own provider).
        const query = step.byVk === false
          ? [{ key: "providers", value: nameOf(step.providerRef) }, { key: "limit", value: "10" }]
          : [{ key: "virtual_key_ids", value: `{{vkid_${sid}}}` }, { key: "limit", value: "10" }];
        items.push(item(nextId("assert-log"), name, request("GET", url(["api", "logs"], query), null),
          events(pollPrerequest(step.waitSeconds), pollTest(name, logAssertLines(sid, step, jsNameOf), cleanupTarget))));
        break;
      }
      case "keyDistribution": {
        const n = step.n || 8;
        for (let i = 0; i < n; i++) {
          const sname = uniq(`sample ${i + 1}/${n} (${step.model})`);
          const body = {
            model: `${nameOf(step.routeRef)}/${step.model}`,
            messages: [{ role: "user", content: "Reply with the single word: ok." }],
            max_tokens: 5,
          };
          items.push(item(nextId("dist-sample"), sname, request("POST", url(["v1", "chat", "completions"]), body),
            events(null, distSampleTest(sname, sid, cleanupTarget, i === 0))));
        }
        const aname = uniq(step.label);
        const assertExec = [
          `var cleanupReq = ${JSON.stringify(cleanupTarget)};`,
          "var ok = true, errMsg = '';",
          "try {",
          ...distAssertLines(sid, step).map((l) => "  " + l),
          "} catch (e) { ok = false; errMsg = e.message; }",
          `pm.test(${JSON.stringify(aname)}, function () { if (!ok) throw new Error(errMsg); });`,
          "if (!ok) { pm.execution.setNextRequest(cleanupReq); }",
        ];
        items.push(item(nextId("dist-assert"), aname, request("GET", url(["health"]), null),
          events(null, assertExec)));
        break;
      }
      case "providerDistribution": {
        const n = step.n || 8;
        for (let i = 0; i < n; i++) {
          const sname = uniq(`sample ${i + 1}/${n} (${step.model})`);
          const headers = step.useVk === false ? [] : [{ key: "x-bf-vk", value: `{{vkval_${sid}}}` }];
          const body = {
            model: step.bareModel ? step.model : `${nameOf(step.routeRef)}/${step.model}`,
            messages: [{ role: "user", content: "Reply with the single word: ok." }],
            max_tokens: 5,
          };
          items.push(item(nextId("pdist-sample"), sname, request("POST", url(["v1", "chat", "completions"]), body, headers),
            events(null, distSampleProviderTest(sname, sid, cleanupTarget, i === 0))));
        }
        const aname = uniq(step.label);
        const only = (step.expectOnly || []).map((r) => jsNameOf(r));
        const never = (step.expectNever || []).map((r) => jsNameOf(r));
        const all = (step.expectAll || []).map((r) => jsNameOf(r));
        const exec = [
          `var cleanupReq = ${JSON.stringify(cleanupTarget)};`,
          "var ok = true, errMsg = '';",
          "try {",
          `  var seen = (pm.collectionVariables.get('pdist_${sid}') || '').split(',').filter(Boolean);`,
          `  var only = [${only.join(", ")}];`,
          `  var never = [${never.join(", ")}];`,
          `  var all = [${all.join(", ")}];`,
          "  if (seen.length === 0) throw new Error('no provider observed across samples');",
          "  seen.forEach(function (p) {",
          "    if (only.length && only.indexOf(p) < 0) throw new Error('provider ' + p + ' served but not in expectOnly ' + JSON.stringify(only));",
          "    if (never.indexOf(p) >= 0) throw new Error('provider ' + p + ' served but was expectNever; observed ' + JSON.stringify(seen));",
          "  });",
          "  all.forEach(function (p) { if (seen.indexOf(p) < 0) throw new Error('provider ' + p + ' expected to serve but did not; observed ' + JSON.stringify(seen)); });",
          "} catch (e) { ok = false; errMsg = e.message; }",
          `pm.test(${JSON.stringify(aname)}, function () { if (!ok) throw new Error(errMsg); });`,
          "if (!ok) { pm.execution.setNextRequest(cleanupReq); }",
        ];
        items.push(item(nextId("pdist-assert"), aname, request("GET", url(["health"]), null), events(null, exec)));
        break;
      }
      case "assertRoutingTrail": {
        // Step 1: poll the log list for the VK's most recent row and capture its id. With
        // fromRequest the row is the last requestId route's own, read by its id below.
        if (!step.fromRequest) {
          const capName = uniq("capture routing log id");
          const capQuery = [{ key: "virtual_key_ids", value: `{{vkid_${sid}}}` }, { key: "limit", value: "1" }];
          const capAssert = [
            "if (pm.response.code !== 200) { throw new Error('logs status ' + pm.response.code); }",
            "var logs = (pm.response.json() || {}).logs || [];",
            "if (!logs.length || !logs[0].id) { throw new Error('no log row yet'); }",
            `pm.collectionVariables.set('logid_${sid}', logs[0].id);`,
          ];
          items.push(item(nextId("capture-log"), capName, request("GET", url(["api", "logs"], capQuery), null),
            events(pollPrerequest(step.waitSeconds), pollTest(capName, capAssert, cleanupTarget))));
        }
        const logIdVar = step.fromRequest ? `reqid_${sid}` : `logid_${sid}`;
        // Step 2: fetch the log detail and assert the routing-engine decision trail.
        const trailName = uniq(step.label);
        const trailAssert = [
          "if (pm.response.code !== 200) { throw new Error('log detail status ' + pm.response.code); }",
          "var row = pm.response.json() || {};",
          "var trail = row.routing_engine_logs || '';",
          `var expected = ${JSON.stringify(step.expectSubstrings || [])}.map(function (s) { return pm.variables.replaceIn(s); });`,
          "expected.forEach(function (s) {",
          "  if (String(trail).indexOf(s) < 0) { throw new Error('routing_engine_logs missing ' + JSON.stringify(s) + '; got ' + JSON.stringify(trail)); }",
          "});",
          `var absent = ${JSON.stringify(step.expectAbsent || [])}.map(function (s) { return pm.variables.replaceIn(s); });`,
          "absent.forEach(function (s) {",
          "  if (String(trail).indexOf(s) >= 0) { throw new Error('routing_engine_logs should not contain ' + JSON.stringify(s) + '; got ' + JSON.stringify(trail)); }",
          "});",
        ];
        // expectCounts: how many times a substring appears, for what must happen once.
        if (step.expectCounts) {
          trailAssert.push(
            `var counts = ${JSON.stringify(step.expectCounts)};`,
            "counts.forEach(function (c) {",
            "  var s = pm.variables.replaceIn(c.substring);",
            "  var n = String(trail).split(s).length - 1;",
            "  if (n !== c.count) { throw new Error('routing_engine_logs has ' + JSON.stringify(s) + ' ' + n + ' times, want ' + c.count + '; got ' + JSON.stringify(trail)); }",
            "});",
          );
        }
        items.push(item(nextId("assert-trail"), trailName,
          request("GET", url(["api", "logs", `{{${logIdVar}}}`]), null),
          events(step.fromRequest ? pollPrerequest(step.waitSeconds) : null, pollTest(trailName, trailAssert, cleanupTarget))));
        break;
      }
      case "cleanup":
        break;
      default:
        throw new Error("unknown step type: " + step.type);
    }
  }

  const cleanupItems = [];
  for (const ref of routerRefs) {
    cleanupItems.push(item(`rt-${sid}-cleanup-router-${ref}`, cleanupRouter(ref),
      request("DELETE", url(["api", "routing", "model-routers", routerIdSeg(ref)]), null),
      events(null, cleanupTest(cleanupRouter(ref)))));
  }
  if (hasVk) {
    cleanupItems.push(item(`rt-${sid}-cleanup-vk`, cleanupVk,
      request("DELETE", url(["api", "governance", "virtual-keys", `{{vkid_${sid}}}`]), null),
      events(null, cleanupTest(cleanupVk))));
  }
  // Delete every provider the scenario created (the primary plus any extra refs).
  // The primary keeps the canonical cleanup name so failing-step jumps still hit it.
  const refsToClean = providerRefs.length ? providerRefs : ["self"];
  for (const ref of refsToClean) {
    const cname = ref === "self" ? cleanupProvider : `cleanup: delete provider ${ref} [${sid}]`;
    cleanupItems.push(item(`rt-${sid}-cleanup-provider-${ref}`, cname,
      request("DELETE", url(["api", "providers", nameOf(ref)]), null),
      events(null, cleanupTest(cname))));
  }

  return {
    name: sc.title,
    description: sc.description,
    item: [...items, { name: "Cleanup", item: cleanupItems }],
    event: events(folderPrerequest([]), null),
  };
}

// --------------------------------------------------------------------------- //
// Scenarios — governance × model-catalog routing
// --------------------------------------------------------------------------- //

// The classifier settings both classifier routers of router-same-classifier-settings are given:
// the llm classifier on the scenario's own provider, with one phrase a tier. One function, so the
// two routers cannot drift apart.
const sameClassifierSettings = (nameOf) => ({
  classifier: "llm",
  llm: { provider: nameOf("self"), model: MODEL_B },
  keywords: { simple_keywords: ["what time is it"], medium_keywords: ["summarise this paragraph"], complex_keywords: ["design a distributed cache"] },
});

const SCENARIOS = [
  {
    id: "vk-allows-model",
    title: "VK allows model, key allows — request routes",
    description: "A VK whose allowed_models includes the request, over a key that allows it, routes successfully and records the key.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: [MODEL_B] })] },
      { type: "route", model: MODEL_B, expectStatus: 200, expectKeyId: "k1", label: "allowed model routes (routing_info.key)" },
      { type: "assertLog", model: MODEL_B, expectStatus: "success", expectSelectedKeyId: "k1", expectVkPresent: true, waitSeconds: 2, label: "log records VK + selected key" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-key-restriction",
    title: "VK key restriction pins routing to the allowed key",
    description: "A VK whose key_ids names only k1 routes through k1 even though k2 also serves the model; routing_info.key and the log's selected_key_name are k1.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] }), key({ id: "k2", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ keyIds: ["k1"], allowedModels: ["*"] })] },
      { type: "route", model: MODEL_B, expectStatus: 200, expectKeyId: "k1", label: "routes via the VK-permitted key (routing_info.key=k1)" },
      { type: "assertLog", model: MODEL_B, expectStatus: "success", expectSelectedKeyId: "k1", expectVkPresent: true, waitSeconds: 2, label: "log selected_key_name is k1" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-disabled-key",
    title: "VK pinned to a disabled key is unroutable even with an enabled sibling",
    description: "The VK restricts to k1, which is disabled; k2 is enabled but VK-excluded, so key selection finds no usable key and rejects with 400.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"], enabled: false }), key({ id: "k2", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ keyIds: ["k1"], allowedModels: ["*"] })] },
      { type: "route", model: MODEL_B, expectStatus: 400, expectErrorSubstr: "no keys found", label: "disabled+VK-restricted key yields no usable key (400)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "multi-key-distribution",
    title: "Both keys serve under weighted key selection",
    description: "Two enabled, equal-weight keys on one provider (no VK). Over a batch of requests, core's weighted key selection routes through both keys.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"], weight: 1 }), key({ id: "k2", models: ["*"], weight: 1 })] },
      { type: "keyDistribution", model: MODEL_B, n: 24, expectKeyIds: ["k1", "k2"], label: "both keys served across 24 samples" },
      { type: "cleanup" },
    ],
  },
  {
    id: "weightless-vk-allowlist-via-logs",
    title: "A weightless VK is an allow-list (no LB), confirmed by the routing log trail",
    description: "Two providers on a VK with NO weights: A's key serves gpt-4o-mini, B's serves only gpt-4o. Routing the bare gpt-4o-mini, governance filters by capability (excludes B) and — having no weighted configs — skips load balancing, routing to A. The routing_engine_logs record the allow-list decisions.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: [MODEL_B] })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: [MODEL_A] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: null, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: null, allowedModels: ["*"] })] },
      { type: "route", model: MODEL_B, bareModel: true, expectStatus: 200, expectProviderOneOf: ["self"], label: "bare model routes to the only capable provider (allow-list, no LB)" },
      { type: "assertRoutingTrail", expectSubstrings: ["excluded: model gpt-4o-mini is not permitted", "No weighted providers", "skipping load balancing"], waitSeconds: 2, label: "log trail shows allow-list filtering and LB skipped" },
      { type: "cleanup" },
    ],
  },
  {
    id: "governance-lb-distributes",
    title: "Governance load-balances a bare model across VK providers",
    description: "A VK with two weighted providers, routing a bare (un-prefixed) model, has governance pick one of them; the log detail's routing_engine_logs records the load-balancing decision.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["*"], weight: 1 })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["*"] }), vkProvider({ providerRef: "b", allowedModels: ["*"] })] },
      { type: "route", model: MODEL_B, bareModel: true, expectStatus: 200, expectProviderOneOf: ["self", "b"], label: "bare model routes via a governance-selected provider" },
      { type: "assertRoutingTrail", expectSubstrings: ["Load balancing model", "Selected provider"], waitSeconds: 2, label: "log detail records the LB decision trail" },
      { type: "cleanup" },
    ],
  },
  {
    id: "reverse-alias-gate",
    title: "Routing the resolved model directly (not the alias) is rejected",
    description: "A key gates the alias name, not the resolved id. Routing the alias resolves and succeeds; routing the resolved model id directly fails the gate — alias targets are not auto-added to the key's Models.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["catwiring-alias-{{run_id}}"], aliases: { "catwiring-alias-{{run_id}}": MODEL_B } })] },
      { type: "route", model: "catwiring-alias-{{run_id}}", useVk: false, expectStatus: 200, expectResolvedModelId: MODEL_B, label: "alias routes (resolves to the model id)" },
      { type: "route", model: MODEL_B, useVk: false, expectStatus: 400, expectErrorSubstr: "no keys found that support model", label: "resolved id routed directly → 400 (not in Models)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-collision-both-keys-serve",
    title: "An alias defined on two keys is served by both, each resolving its own target",
    description: "Two keys of a standard openai provider define the same alias, k1 to gpt-4o and k2 to gpt-4o-mini. Both keys pass the key gate for the alias, so a request pinned to either one is served by it, and each resolves the alias to its own target. The catalog's alias index, used for listing and pricing, takes the last enabled key instead: /api/models/details prices the alias at k2's gpt-4o-mini, and at k1's dearer gpt-4o once k2 is disabled. A standard provider is used because only datasheet providers carry prices. Serial-only (global standard provider).",
    steps: [
      { type: "addProvider", providerKind: "standard", providerType: "openai", keys: [
        key({ id: "k1", models: ["catwiring-dup-{{run_id}}"], aliases: { "catwiring-dup-{{run_id}}": MODEL_A } }),
        key({ id: "k2", models: ["catwiring-dup-{{run_id}}"], aliases: { "catwiring-dup-{{run_id}}": MODEL_B } }),
      ] },
      { type: "route", model: "catwiring-dup-{{run_id}}", useVk: false, pinKey: "k1", expectStatus: 200, expectKeyId: "k1", expectResolvedModelId: MODEL_A, label: "k1 serves the alias, resolving it to its own target (gpt-4o)" },
      { type: "route", model: "catwiring-dup-{{run_id}}", useVk: false, pinKey: "k2", expectStatus: 200, expectKeyId: "k2", expectResolvedModelId: MODEL_B, label: "k2 serves the alias, resolving it to its own target (gpt-4o-mini)" },
      { type: "aliasPricing", alias: "catwiring-dup-{{run_id}}", as: "both", label: "the catalog prices the alias from the last enabled key (k2)" },
      { type: "updateKey", key: key({ id: "k2", models: ["catwiring-dup-{{run_id}}"], aliases: { "catwiring-dup-{{run_id}}": MODEL_B }, enabled: false }), label: "disable k2" },
      { type: "aliasPricing", alias: "catwiring-dup-{{run_id}}", as: "k1only", higherThan: "both", label: "with k2 disabled the alias is priced from k1's dearer gpt-4o, so the price with both keys was k2's gpt-4o-mini" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-case-insensitive-fallback",
    title: "A request resolves to an alias whose name differs only in case",
    description: "A key defines a mixed-case alias and gates the mixed-case name. Routing the lowercased form finds no exact-case alias but resolves through a case-insensitive fallback to the same target.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["CatWiring-CI-{{run_id}}"], aliases: { "CatWiring-CI-{{run_id}}": MODEL_B } })] },
      { type: "route", model: "catwiring-ci-{{run_id}}", useVk: false, expectStatus: 200, expectKeyId: "k1", expectResolvedModelId: MODEL_B, label: "lowercased request resolves via case-insensitive fallback" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-routing-no-vk",
    title: "Alias resolves at routing with no governance",
    description: "Pure model-catalog case (no VK): a key alias routes an inference request to the underlying model; routing_info.resolved_key_alias records the resolution.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["catwiring-alias-{{run_id}}"], aliases: { "catwiring-alias-{{run_id}}": MODEL_B } })] },
      { type: "route", model: "catwiring-alias-{{run_id}}", useVk: false, expectStatus: 200, expectResolvedModelId: MODEL_B, label: "alias routes to underlying model (no VK)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "blacklist-gate-no-vk",
    title: "Key blacklist gates routing with no governance",
    description: "Pure model-catalog case (no VK): a key that allows all models but blacklists one rejects that model at key selection.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"], blacklisted: [MODEL_A] })] },
      { type: "route", model: MODEL_A, useVk: false, expectStatus: 400, expectErrorSubstr: "no keys found that support model", label: "blacklisted model rejected by key gate (400)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "disabled-key-gate-no-vk",
    title: "Disabled key makes a model unroutable with no governance",
    description: "Pure model-catalog case (no VK): the only key for a model is disabled, so key selection finds nothing.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"], enabled: false })] },
      { type: "route", model: MODEL_B, useVk: false, expectStatus: 400, expectErrorSubstr: "no keys found", label: "disabled key yields no route (400)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "route-azure",
    title: "Azure provider routes a deployment",
    description: "A standard azure provider (value + azure_key_config via env.) routes a model whose deployment matches the name. Serial-only (global standard provider).",
    steps: [
      { type: "addProvider", providerType: "azure", keys: [key({ id: "kaz", models: ["*"] })] },
      { type: "route", model: PROVIDERS.azure.model, useVk: false, expectStatus: 200, expectKeyId: "kaz", label: "azure routes its deployment (200)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "route-bedrock",
    title: "Bedrock routes Claude via a key alias to its inference profile",
    description: "A standard bedrock provider whose key aliases the common name claude-sonnet-4-5 to the cross-region inference-profile id. Routing the friendly name resolves to the wire id. Serial-only (global standard provider).",
    steps: [
      { type: "addProvider", providerType: "bedrock", keys: [key({ id: "kbr", models: ["*"] })] },
      { type: "route", model: PROVIDERS.bedrock.model, useVk: false, expectStatus: 200, expectKeyId: "kbr", expectResolvedModelId: "us.anthropic.claude-sonnet-4-5-20250929-v1:0", label: "bedrock alias resolves to inference profile (200)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "cross-provider-same-model-openai-azure",
    title: "Governance LBs one openai model across OpenAI and Azure",
    description: "OpenAI (custom) and Azure (standard) both serve gpt-4o-mini. A VK over both, routing the bare model, has governance distribute across them. Serial-only (Azure is a global standard provider).",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ko", models: ["*"], weight: 1 })] },
      { type: "addProvider", ref: "az", providerType: "azure", keys: [key({ id: "kaz", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["*"] }), vkProvider({ providerRef: "az", allowedModels: ["*"] })] },
      { type: "route", model: "gpt-4o-mini", bareModel: true, expectStatus: 200, expectProviderOneOf: ["self", "az"], label: "bare model routes via openai or azure (200)" },
      { type: "assertRoutingTrail", expectSubstrings: ["Load balancing model gpt-4o-mini", "Selected provider"], waitSeconds: 2, label: "log detail records openai/azure LB trail" },
      { type: "cleanup" },
    ],
  },
  {
    id: "route-anthropic",
    title: "Standard anthropic provider routes its model",
    description: "A standard anthropic provider (key via env.) routes a claude model. Serial-only (global standard provider).",
    steps: [
      { type: "addProvider", providerKind: "standard", providerType: "anthropic", keys: [key({ id: "ka", models: ["*"] })] },
      { type: "route", model: PROVIDERS.anthropic.model, useVk: false, expectStatus: 200, expectKeyId: "ka", label: "anthropic provider routes claude model (200)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "route-gemini",
    title: "Standard gemini provider routes its model",
    description: "A standard gemini provider (key via env.) routes a gemini model. Serial-only (global standard provider).",
    steps: [
      { type: "addProvider", providerKind: "standard", providerType: "gemini", keys: [key({ id: "kg", models: ["*"] })] },
      { type: "route", model: PROVIDERS.gemini.model, useVk: false, expectStatus: 200, expectKeyId: "kg", label: "gemini provider routes gemini model (200)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-explicit-allowed-key-cant-serve",
    title: "VK explicit allowed-model the key can't serve → 400 (not 403)",
    description: "Unlike the wildcard case (which 403s via the catalog-aware check), an EXPLICIT allowed_models entry is string-matched by governance and passes; key selection then fails → 400 no keys. Pins the explicit-vs-wildcard split: explicit lists are string-matched by governance, wildcards go through the catalog-aware check.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: [MODEL_B] })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: [MODEL_A] })] },
      { type: "route", model: MODEL_A, expectStatus: 400, expectErrorSubstr: "no keys found that support model", label: "explicit allowed model the key can't serve → 400" },
      { type: "cleanup" },
    ],
  },
  {
    id: "key-gate-beats-key-weight",
    title: "Within a provider, the capable key wins over a higher-weight incapable key",
    description: "Pure model-catalog/core (no VK): k1 weight 99 allows only gpt-4o; k2 weight 1 allows gpt-4o-mini. Routing gpt-4o-mini always uses k2 — key capability filters before weighting.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: [MODEL_A], weight: 99 }), key({ id: "k2", models: [MODEL_B], weight: 1 })] },
      { type: "keyDistribution", model: MODEL_B, n: 10, expectOnly: ["k2"], expectNever: ["k1"], label: "all requests use the capable 1%-weight key (k2)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "key-blacklist-intersection",
    title: "A model blacklisted on one key still routes via a sibling key",
    description: "Pure model-catalog/core (no VK): k1 allows all but blacklists gpt-4o-mini; k2 allows all. The model is blocked only on k1, so the provider still serves it via k2 (blacklist is per-key, not provider-wide unless all keys block).",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"], blacklisted: [MODEL_B] }), key({ id: "k2", models: ["*"] })] },
      { type: "keyDistribution", model: MODEL_B, n: 10, expectOnly: ["k2"], expectNever: ["k1"], label: "model routes via the non-blacklisting sibling (k2)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "lb-partial-exclusion-3-providers",
    title: "LB excludes only the incapable provider; the rest still split",
    description: "Three providers on a VK: A can't serve the model (key gate), B and C can. Routing the bare model, A is excluded and B/C both still serve over the batch.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: [MODEL_A], weight: 1 })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"], weight: 1 })] },
      { type: "addProvider", ref: "c", providerType: "openai", keys: [key({ id: "kc", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 1, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: ["*"] }), vkProvider({ providerRef: "c", weight: 1, allowedModels: ["*"] })] },
      { type: "providerDistribution", model: MODEL_B, bareModel: true, n: 12, expectOnly: ["b", "c"], expectAll: ["b", "c"], expectNever: ["self"], label: "incapable provider excluded; B and C both serve" },
      { type: "cleanup" },
    ],
  },
  {
    id: "lb-skips-model-gated-provider",
    title: "LB skips a 99%-weight provider that can't serve the model (key gate)",
    description: "A is weighted 99% but its key only allows gpt-4o; B is weighted 1% and allows all. Routing the bare gpt-4o-mini, capability filtering excludes A entirely, so every request lands on B regardless of weight.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["gpt-4o"], weight: 1 })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 99, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: ["*"] })] },
      { type: "providerDistribution", model: "gpt-4o-mini", bareModel: true, n: 10, expectOnly: ["b"], expectNever: ["self"], label: "all requests go to the 1% provider (99% provider can't serve the model)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "lb-skips-blacklisted-provider",
    title: "LB skips a 99%-weight provider that blacklists the model",
    description: "A (99%) allows all models but blacklists gpt-4o-mini on its key; B (1%) allows all. The blacklist removes A from the candidates, so every request lands on B.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["*"], blacklisted: ["gpt-4o-mini"], weight: 1 })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 99, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: ["*"] })] },
      { type: "providerDistribution", model: "gpt-4o-mini", bareModel: true, n: 10, expectOnly: ["b"], expectNever: ["self"], label: "all requests go to the 1% provider (99% provider blacklists the model)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "lb-skips-disabled-key-provider",
    title: "LB skips a 99%-weight provider whose only key is disabled",
    description: "A (99%) has its only key disabled; B (1%) is enabled. With no usable key, A is excluded, so every request lands on B.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["*"], enabled: false, weight: 1 })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 99, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: ["*"] })] },
      { type: "providerDistribution", model: "gpt-4o-mini", bareModel: true, n: 10, expectOnly: ["b"], expectNever: ["self"], label: "all requests go to the 1% provider (99% provider key disabled)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "cross-provider-allowlist",
    title: "VK provider allowlist blocks a different real provider",
    description: "A VK that lists only the openai provider routes openai, and governance refuses an explicit request to the anthropic provider with 403 provider_blocked.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ko", models: ["*"] })] },
      { type: "addProvider", ref: "b", providerType: "anthropic", keys: [key({ id: "ka", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["*"] })] },
      { type: "route", routeRef: "self", model: PROVIDERS.openai.model, expectStatus: 200, expectKeyId: "ko", label: "VK-allowed provider routes (200)" },
      { type: "route", routeRef: "b", model: PROVIDERS.anthropic.model, expectStatus: 403, expectErrorType: "provider_blocked", expectErrorSubstr: "is not allowed", label: "VK-disallowed provider blocked (403 provider_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "cross-provider-same-model-lb",
    title: "Governance LBs one Claude model across Anthropic, Vertex, and Bedrock",
    description: "Anthropic (custom, native), Vertex (standard, native) and Bedrock (standard, via a key alias to its inference-profile id) all serve claude-sonnet-4-5. A VK over all three, routing the bare model, has governance distribute across the heterogeneous providers; the log detail records the LB trail. Serial-only (Vertex/Bedrock are global standard providers).",
    steps: [
      { type: "addProvider", ref: "self", providerType: "anthropic", keys: [key({ id: "kan", models: ["*"], weight: 1 })] },
      { type: "addProvider", ref: "vtx", providerType: "vertex", keys: [key({ id: "kvx", models: ["*"], weight: 1 })] },
      { type: "addProvider", ref: "bdr", providerType: "bedrock", keys: [key({ id: "kbd", models: ["*"], weight: 1 })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["*"] }), vkProvider({ providerRef: "vtx", allowedModels: ["*"] }), vkProvider({ providerRef: "bdr", allowedModels: ["*"] })] },
      { type: "route", model: "claude-sonnet-4-5", bareModel: true, expectStatus: 200, expectProviderOneOf: ["self", "vtx", "bdr"], label: "bare claude model routes via anthropic, vertex, or bedrock (200)" },
      { type: "assertRoutingTrail", expectSubstrings: ["Load balancing model claude-sonnet-4-5", "Selected provider"], waitSeconds: 2, label: "log detail records cross-provider LB trail" },
      { type: "cleanup" },
    ],
  },
  {
    id: "fallback-cross-provider",
    title: "A request fails over to a healthy provider via a request-level fallback",
    description: "The primary provider's key is invalid, so its attempt fails upstream; a request-level fallback to a second provider serving the same model succeeds. routing_info marks the fallback and records the original primary provider/model. The /v1 endpoint takes fallbacks in the string \"provider/model\" form.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad", models: [MODEL_B], badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kgood", models: [MODEL_B] })] },
      { type: "route", model: MODEL_B, routeRef: "self", useVk: false, fallbacks: [{ providerRef: "b", model: MODEL_B }], expectStatus: 200, expectProviderOneOf: ["b"], expectIsFallback: true, expectPrimaryProviderRef: "self", expectPrimaryModel: MODEL_B, label: "primary fails; request fallback serves (200, is_fallback)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "fallback-chain-first-healthy-wins",
    title: "Fallbacks are tried in order until one succeeds",
    description: "The primary and the first fallback both have invalid keys; the second fallback succeeds. routing_info reports the surviving provider and still names the original primary.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad1", models: [MODEL_B], badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kbad2", models: [MODEL_B], badKey: true })] },
      { type: "addProvider", ref: "c", keys: [key({ id: "kgood", models: [MODEL_B] })] },
      { type: "route", model: MODEL_B, routeRef: "self", useVk: false, fallbacks: [{ providerRef: "b", model: MODEL_B }, { providerRef: "c", model: MODEL_B }], expectStatus: 200, expectProviderOneOf: ["c"], expectIsFallback: true, expectPrimaryProviderRef: "self", expectPrimaryModel: MODEL_B, label: "chain falls through to the only healthy provider (200)" },
      // Every attempt of the chain is its own log row: the primary failed at fallback_index 0, B was
      // tried and failed at index 1, and C served at index 2.
      { type: "assertLog", byVk: false, providerRef: "self", model: MODEL_B, expectStatus: "error", expectFallbackIndex: 0, waitSeconds: 2, label: "log: the primary failed (fallback_index 0)" },
      { type: "assertLog", byVk: false, providerRef: "b", model: MODEL_B, expectStatus: "error", expectFallbackIndex: 1, waitSeconds: 0, label: "log: B was tried and failed (fallback_index 1)" },
      { type: "assertLog", byVk: false, providerRef: "c", model: MODEL_B, expectStatus: "success", expectFallbackIndex: 2, waitSeconds: 0, label: "log: C served (fallback_index 2)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "fallback-pruned-by-vk-allowlist",
    title: "A fallback to a provider off the VK allowlist is pruned, not tried",
    description: "Under a VK that permits only the primary provider, a request-level fallback to a healthy off-allowlist provider is pruned before the attempt loop. The primary's invalid key fails with a 401 and the pruned provider — which has a valid key and would otherwise return 200 — never rescues it, proving it was dropped.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad", models: [MODEL_B], badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kgood", models: [MODEL_B] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["*"] })] },
      { type: "route", model: MODEL_B, routeRef: "self", fallbacks: [{ providerRef: "b", model: MODEL_B }], expectStatus: 401, expectErrorSubstr: "Incorrect API key", label: "off-allowlist fallback pruned; request fails on the primary (401)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "exhausted-fallback-headers",
    title: "An exhausted chain returns the primary's error and headers, on /v1 and /openai",
    description: "The primary is an OpenAI-backed provider asked for a model that does not exist; its fallback is an Anthropic-backed provider asked for another missing model. Both fail, so the caller gets OpenAI's 404, the routing-info provider header names the primary, and no anthropic-* header from the fallback leaks into the response. The fallback's own log row, under the request as its parent, shows it was tried. Run on /v1/chat/completions and on the /openai drop-in route.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ko", models: ["*"] })] },
      { type: "addProvider", ref: "b", providerType: "anthropic", keys: [key({ id: "ka", models: ["*"] })] },
      { type: "route", model: "gpt-harness-missing-model", routeRef: "self", useVk: false, requestId: true, fallbacks: [{ providerRef: "b", model: "claude-harness-missing-model" }], expectStatus: 404, expectErrorSubstr: "gpt-harness-missing-model", expectHeaderProviderRef: "self", expectNoHeaderPrefix: "anthropic-", label: "/v1: both fail; OpenAI's 404 with the primary's headers and no anthropic-* header" },
      { type: "assertFallbackRow", providerRef: "b", fallbackIndex: 1, expectStatus: "error", model: "claude-harness-missing-model", waitSeconds: 2, label: "/v1: the Anthropic fallback was tried and failed (fallback_index 1)" },
      { type: "route", endpoint: "openai", model: "gpt-harness-missing-model", routeRef: "self", useVk: false, requestId: true, fallbacks: [{ providerRef: "b", model: "claude-harness-missing-model" }], expectStatus: 404, expectErrorSubstr: "gpt-harness-missing-model", expectHeaderProviderRef: "self", expectNoHeaderPrefix: "anthropic-", label: "/openai: both fail; OpenAI's 404 with the primary's headers and no anthropic-* header" },
      { type: "assertFallbackRow", providerRef: "b", fallbackIndex: 1, expectStatus: "error", model: "claude-harness-missing-model", waitSeconds: 2, label: "/openai: the Anthropic fallback was tried and failed (fallback_index 1)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "anthropic-route-catalog-candidates",
    title: "The /anthropic route resolves a model Anthropic does not serve through the catalog",
    description: "Two run-scoped OpenAI-backed providers serve a run-scoped alias of gpt-4o-mini; Anthropic does not. A bare request for the alias on /anthropic/v1/messages is resolved by the model catalog to one of the two, the trail lists exactly those two as the candidates (so Anthropic is not one), and the answer is Messages-shaped.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["catwiring-mce04-{{run_id}}"], aliases: { "catwiring-mce04-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["catwiring-mce04-{{run_id}}"], aliases: { "catwiring-mce04-{{run_id}}": MODEL_B } })] },
      { type: "route", endpoint: "anthropic", model: "catwiring-mce04-{{run_id}}", bareModel: true, useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["self", "b"], label: "/anthropic resolves the alias to one of the two providers (Messages-shaped 200)" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["[model-catalog]", "No provider specified for model catwiring-mce04-{{run_id}}, found 2 options in model catalog: [", "catwiring-rt-anthropic-route-catalog-candidates-{{run_id}}", "catwiring-rt-anthropic-route-catalog-candidates-b-{{run_id}}"], waitSeconds: 2, label: "trail: the catalog's candidates are exactly the two providers, not Anthropic" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-fallback-to-deleted-provider",
    title: "A router fallback whose provider was deleted is skipped with a trail warning",
    description: "A conditional router matching this run's own alias targets A, whose key OpenAI refuses, with fallbacks [C, B]. C is deleted after the router is saved. The request's trail warns that the C fallback does not name a known provider, and B serves as fallback 1.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad", models: ["catwiring-rr16-{{run_id}}"], aliases: { "catwiring-rr16-{{run_id}}": MODEL_B }, badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-rr16-{{run_id}}"], aliases: { "catwiring-rr16-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "c", keys: [key({ id: "kc", models: ["catwiring-rr16-{{run_id}}"], aliases: { "catwiring-rr16-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", cel: "model == 'catwiring-rr16-{{run_id}}'", targets: [{ providerRef: "self", model: "catwiring-rr16-{{run_id}}" }], fallbacks: [{ providerRef: "c", model: "catwiring-rr16-{{run_id}}" }, { providerRef: "b", model: "catwiring-rr16-{{run_id}}" }], label: "create the router: A, fallbacks [C, B]" },
      { type: "deleteProvider", ref: "c", label: "delete provider C after saving the router" },
      { type: "route", model: "catwiring-rr16-{{run_id}}", bareModel: true, useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], expectIsFallback: true, expectPrimaryProviderRef: "self", label: "the router's A fails; B serves" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["fallback \"catwiring-rt-router-fallback-to-deleted-provider-c-{{run_id}}/catwiring-rr16-{{run_id}}\" skipped: it does not name a known provider"], waitSeconds: 2, label: "trail: the deleted C fallback is skipped with a warning" },
      { type: "assertFallbackRow", providerRef: "b", fallbackIndex: 1, expectStatus: "success", waitSeconds: 0, label: "B served as fallback 1" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-auto-attached-fallback",
    title: "A VK with weighted providers auto-attaches the others as fallbacks",
    description: "A VK weights two providers that both serve the model; the dominant-weight provider's key is invalid. With no request-level fallbacks, governance auto-attaches the remaining weighted config as a fallback, so every request still lands on the healthy low-weight provider — whether it was the load-balanced primary or the fallback.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad", models: [MODEL_B], badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kgood", models: [MODEL_B] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 100, allowedModels: ["*"] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: ["*"] })] },
      { type: "providerDistribution", model: MODEL_B, bareModel: true, n: 6, expectOnly: ["b"], label: "every request lands on the healthy provider via the auto-attached fallback" },
      { type: "cleanup" },
    ],
  },
  {
    id: "standard-openai-route",
    title: "Standard (non-custom) openai provider routes",
    description: "A standard openai provider created via the API routes its model. The catalog is datasheet-backed (no live-cache wait). NOTE: standard providers are global singletons — this scenario is NOT run-id-isolated; run serially against a clean instance, not in parallel shards.",
    steps: [
      { type: "addProvider", providerKind: "standard", providerType: "openai", keys: [key({ id: "ko", models: ["gpt-4o-mini"] })] },
      { type: "route", model: "gpt-4o-mini", useVk: false, expectStatus: 200, expectKeyId: "ko", label: "standard openai routes its model (200)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "standard-openai-vk-gate",
    title: "Governance gates a standard provider",
    description: "A VK over a standard openai provider routes an allowed model, and governance refuses a disallowed one with 403 model_blocked. Serial-only (global provider).",
    steps: [
      { type: "addProvider", providerKind: "standard", providerType: "openai", keys: [key({ id: "ko", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", allowedModels: ["gpt-4o-mini"] })] },
      { type: "route", model: "gpt-4o-mini", expectStatus: 200, expectKeyId: "ko", label: "VK-allowed model routes on standard provider (200)" },
      { type: "assertLog", model: "gpt-4o-mini", expectStatus: "success", expectSelectedKeyId: "ko", expectVkPresent: true, waitSeconds: 2, label: "log records standard-provider route" },
      { type: "route", model: "gpt-4o", expectStatus: 403, expectErrorType: "model_blocked", expectErrorSubstr: "is not allowed", label: "VK-disallowed model refused (403 model_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-allowed-models-exclude-provider",
    title: "A VK's allowed_models exclude a provider from the weighted pick",
    description: "Two equally weighted providers whose keys serve every model; the VK lets A serve only gpt-4o-mini and B only gpt-4o. A bare gpt-4o-mini goes to A every time, the trail names B as excluded because the model is not permitted, the pick runs over A alone, and no fallback is attached.",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["*"] })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 1, allowedModels: [MODEL_B] }), vkProvider({ providerRef: "b", weight: 1, allowedModels: [MODEL_A] })] },
      { type: "providerDistribution", model: MODEL_B, bareModel: true, n: 10, expectOnly: ["self"], expectNever: ["b"], label: "every request goes to A, the only provider the VK permits the model on" },
      { type: "assertRoutingTrail", expectSubstrings: ["Provider catwiring-rt-vk-allowed-models-exclude-provider-b-{{run_id}} excluded: model gpt-4o-mini is not permitted", "(from 1 weighted:"], expectAbsent: ["fallback providers"], waitSeconds: 2, label: "trail: B excluded by allowed_models, the pick over A alone, no fallbacks attached" },
      { type: "cleanup" },
    ],
  },
  {
    id: "named-provider-skips-load-balancing",
    title: "A provider-prefixed request skips the VK's load balancing, with or without the VK",
    description: "A VK weights A and B equally. A request naming B is served by B as the primary both with the VK and without it; governance logs that it skips load balancing because the provider is already set and attaches no fallbacks, and the model catalog writes nothing to the trail (stored entries read \"[<ts>] [<engine>] [<level>] - <message>\").",
    steps: [
      { type: "addProvider", ref: "self", providerType: "openai", keys: [key({ id: "ka", models: ["*"] })] },
      { type: "addProvider", ref: "b", providerType: "openai", keys: [key({ id: "kb", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ providerRef: "self", weight: 1 }), vkProvider({ providerRef: "b", weight: 1 })] },
      { type: "route", routeRef: "b", model: MODEL_B, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], expectIsFallback: false, label: "B/m with the VK is served by B" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Skipping load balancing for model gpt-4o-mini: provider catwiring-rt-named-provider-skips-load-balancing-b-{{run_id}} already set"], expectAbsent: ["fallback providers", "Selected provider", "[model-catalog]"], waitSeconds: 2, label: "trail (with the VK): governance skips load balancing, attaches nothing, and the catalog writes nothing" },
      { type: "route", routeRef: "b", model: MODEL_B, useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], expectIsFallback: false, label: "B/m without a VK is served by B" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Skipping load balancing for model gpt-4o-mini: provider catwiring-rt-named-provider-skips-load-balancing-b-{{run_id}} already set"], expectAbsent: ["fallback providers", "Selected provider", "[model-catalog]"], waitSeconds: 2, label: "trail (without a VK): governance skips load balancing, attaches nothing, and the catalog writes nothing" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-weight-update-validation",
    title: "Updating a VK provider weight: -1 is refused, 0, 0.5 and null are stored",
    description: "A VK created with weight 1 on its provider is updated through PUT /api/governance/virtual-keys/{id}: weight -1 is refused with 400 and leaves the stored weight at 1; weights 0, 0.5 and null are each accepted with 200 and read back as stored.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ weight: 1 })] },
      { type: "updateVK", providerConfigs: [{ providerRef: "self", weight: -1 }], expectStatus: 400, expectErrorSubstr: "weight", label: "PUT weight -1 is refused (400)" },
      { type: "assertVKWeight", providerRef: "self", expectWeight: 1, label: "the refused update left the weight at 1" },
      { type: "updateVK", providerConfigs: [{ providerRef: "self", weight: 0 }], expectStatus: 200, label: "PUT weight 0 is accepted (200)" },
      { type: "assertVKWeight", providerRef: "self", expectWeight: 0, label: "weight 0 is stored" },
      { type: "updateVK", providerConfigs: [{ providerRef: "self", weight: 0.5 }], expectStatus: 200, label: "PUT weight 0.5 is accepted (200)" },
      { type: "assertVKWeight", providerRef: "self", expectWeight: 0.5, label: "weight 0.5 is stored" },
      { type: "updateVK", providerConfigs: [{ providerRef: "self", weight: null }], expectStatus: 200, label: "PUT weight null is accepted (200)" },
      { type: "assertVKWeight", providerRef: "self", expectWeight: null, label: "weight null is stored (no weight)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-model-whitelist",
    title: "VK model whitelist blocks an unlisted model",
    description: "A model outside the VK's allowed_models is refused by governance with 403 model_blocked.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: [MODEL_B] })] },
      { type: "route", model: MODEL_A, expectStatus: 403, expectErrorType: "model_blocked", expectErrorSubstr: "is not allowed", label: "unlisted model refused (403 model_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-blacklist",
    title: "VK blacklist blocks a model",
    description: "A model in the VK's blacklisted_models is refused by governance with 403 model_blocked.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: ["*"], blacklistedModels: [MODEL_A] })] },
      { type: "route", model: MODEL_A, expectStatus: 403, expectErrorType: "model_blocked", expectErrorSubstr: "is not allowed", label: "blacklisted model refused (403 model_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-wildcard-bounded-by-key-gate",
    title: "VK wildcard is still bounded by the key gate",
    description: "Governance's model check is catalog-aware, so a VK allowed_models=[\"*\"] does not widen past what the key actually gates; a model outside the key's allow-list is blocked by governance.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: [MODEL_B] })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: ["*"] })] },
      { type: "route", model: MODEL_A, expectStatus: 403, expectErrorType: "model_blocked", expectErrorSubstr: "Model 'gpt-4o' is not allowed for virtual key 'catwiring-rtvk-vk-wildcard-bounded-by-key-gate-{{run_id}}'", label: "wildcard VK still blocks a model the key does not gate (403, names the VK)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "catalog-gate-blocks-no-vk",
    title: "Catalog/key gate blocks a model with no VK (400)",
    description: "Without a virtual key, governance does not run; a model the key does not allow is rejected by core key selection.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: [MODEL_B] })] },
      { type: "route", model: MODEL_A, useVk: false, expectStatus: 400, expectErrorSubstr: "no keys found that support model", label: "key gate rejects unlisted model (400)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-whitelisted-by-name",
    title: "Alias whitelisted by name resolves",
    description: "When the VK whitelists the alias name, the request routes and resolves to the underlying model id.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["catwiring-alias-{{run_id}}"], aliases: { "catwiring-alias-{{run_id}}": MODEL_B } })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: ["catwiring-alias-{{run_id}}"] })] },
      { type: "route", model: "catwiring-alias-{{run_id}}", expectStatus: 200, expectResolvedModelId: MODEL_B, label: "alias routes, resolves to model id" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-of-allowed-model",
    title: "Alias of an allowed model resolves",
    description: "The VK whitelists only the resolved model id and the request uses the alias name; every enabled key maps the alias to that allowed model, so governance accepts it (#8528).",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["catwiring-alias-{{run_id}}"], aliases: { "catwiring-alias-{{run_id}}": MODEL_B } })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: [MODEL_B] })] },
      { type: "route", model: "catwiring-alias-{{run_id}}", expectStatus: 200, expectResolvedModelId: MODEL_B, label: "alias of allowed model routes, resolves to model id" },
      { type: "cleanup" },
    ],
  },
  {
    id: "alias-vs-whitelist",
    title: "Alias of an unlisted model is blocked",
    description: "The VK whitelists a different model than the alias resolves to; neither the alias name nor its target is in allowed_models, so governance refuses it with 403 model_blocked.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["catwiring-alias-{{run_id}}"], aliases: { "catwiring-alias-{{run_id}}": MODEL_B } })] },
      { type: "createVK", providerConfigs: [vkProvider({ allowedModels: [MODEL_A] })] },
      { type: "route", model: "catwiring-alias-{{run_id}}", expectStatus: 403, expectErrorType: "model_blocked", expectErrorSubstr: "is not allowed", label: "alias of unlisted model refused (403 model_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "vk-empty-configs",
    title: "VK with no provider configs blocks everything",
    description: "An empty provider_configs is deny-by-default: governance refuses the provider with 403 provider_blocked.",
    steps: [
      { type: "addProvider", keys: [key({ id: "k1", models: ["*"] })] },
      { type: "createVK", providerConfigs: [] },
      { type: "route", model: MODEL_B, expectStatus: 403, expectErrorType: "provider_blocked", expectErrorSubstr: "is not allowed", label: "deny-by-default refuses the request (403 provider_blocked)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-conditional-routes",
    title: "A conditional model router sends the requests its condition takes to its target",
    description: "Two providers serve this run's own model and the caller names provider A. A conditional model router whose condition is that model moves the request to B without classifying it, and the trail records the router's choice. Turned off, the router takes nothing and A serves again.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["catwiring-mr01-{{run_id}}"], aliases: { "catwiring-mr01-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr01-{{run_id}}"], aliases: { "catwiring-mr01-{{run_id}}": MODEL_B } })] },
      { type: "route", model: "catwiring-mr01-{{run_id}}", useVk: false, expectStatus: 200, label: "with no router, the provider the caller named serves" },
      { type: "createRouter", ref: "r1", cel: "model == 'catwiring-mr01-{{run_id}}'", targets: [{ providerRef: "b", model: "catwiring-mr01-{{run_id}}" }], label: "create a conditional router: this run's model goes to B" },
      { type: "route", model: "catwiring-mr01-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], label: "the router moves the request to B" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Model router 'catwiring-rtr-router-conditional-routes-r1-{{run_id}}' → provider=catwiring-rt-router-conditional-routes-b-{{run_id}}, model=catwiring-mr01-{{run_id}}"], expectAbsent: ["Model router 'catwiring-rtr-router-conditional-routes-r1-{{run_id}}' tier="], waitSeconds: 2, label: "trail: the router's choice, made without a tier" },
      { type: "routerApi", ref: "r1", method: "PUT", path: ["routing"], body: { enabled: false }, expectStatus: 200, label: "turn the router off" },
      { type: "route", model: "catwiring-mr01-{{run_id}}", useVk: false, expectStatus: 200, label: "with the router off, the provider the caller named serves again" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-no-target-passes-on",
    title: "A conditional model router with no target passes the request to the next router",
    description: "Two conditional routers take this run's own model. The first has no target, so it passes the request on; the second sends it to B. The trail shows the first router passing the request on and the second choosing.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["catwiring-mr02-{{run_id}}"], aliases: { "catwiring-mr02-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr02-{{run_id}}"], aliases: { "catwiring-mr02-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", ref: "r1", cel: "model == 'catwiring-mr02-{{run_id}}'", priority: 0, label: "create the first router, with no target" },
      { type: "createRouter", ref: "r2", cel: "model == 'catwiring-mr02-{{run_id}}'", priority: 1, targets: [{ providerRef: "b", model: "catwiring-mr02-{{run_id}}" }], label: "create the second router: this run's model goes to B" },
      { type: "route", model: "catwiring-mr02-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], label: "the second router routes the request to B" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Model router 'catwiring-rtr-router-no-target-passes-on-r1-{{run_id}}' has no target; trying the next router", "Model router 'catwiring-rtr-router-no-target-passes-on-r2-{{run_id}}' → provider=catwiring-rt-router-no-target-passes-on-b-{{run_id}}, model=catwiring-mr02-{{run_id}}"], waitSeconds: 2, label: "trail: the first router passes the request on, the second chooses" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-chain",
    title: "A chained model router's choice is walked again by the routers",
    description: "The first router takes this run's first model, is chained, and changes only the model, to this run's second model. The routers are then walked again on that model, and the second router, which takes only the second model and sits ahead of the first, sends it to B. Neither router reaches B on its own: only B serves the second model.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["catwiring-mr03a-{{run_id}}"], aliases: { "catwiring-mr03a-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr03b-{{run_id}}"], aliases: { "catwiring-mr03b-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", ref: "r1", cel: "model == 'catwiring-mr03a-{{run_id}}'", priority: 1, chain: true, targets: [{ model: "catwiring-mr03b-{{run_id}}" }], label: "create the chained router: the first model becomes the second" },
      { type: "createRouter", ref: "r2", cel: "model == 'catwiring-mr03b-{{run_id}}'", priority: 0, targets: [{ providerRef: "b", model: "catwiring-mr03b-{{run_id}}" }], label: "create the second router, ahead of the first: the second model goes to B" },
      { type: "route", model: "catwiring-mr03a-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], label: "a request for the first model is served by B" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["model=catwiring-mr03b-{{run_id}} [chain, continuing]", "Chain step 1: re-evaluating with provider=catwiring-rt-router-chain-{{run_id}}, model=catwiring-mr03b-{{run_id}}", "Model router 'catwiring-rtr-router-chain-r2-{{run_id}}' → provider=catwiring-rt-router-chain-b-{{run_id}}, model=catwiring-mr03b-{{run_id}}"], waitSeconds: 2, label: "trail: the chained choice, the second pass and the second router's choice" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-keeps-model-adds-fallbacks",
    title: "A model router target that names no provider or model keeps the request's and adds its fallbacks",
    description: "Provider A's key is one OpenAI refuses. A conditional router takes this run's own model with a target that names neither a provider nor a model, only a fallback to B. The request stays on A, fails there, and B serves it as the router's fallback.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "kbad", models: ["catwiring-mr04-{{run_id}}"], aliases: { "catwiring-mr04-{{run_id}}": MODEL_B }, badKey: true })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr04-{{run_id}}"], aliases: { "catwiring-mr04-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", ref: "r1", cel: "model == 'catwiring-mr04-{{run_id}}'", targets: [{}], fallbacks: [{ providerRef: "b", model: "catwiring-mr04-{{run_id}}" }], label: "create the router: keep the request's provider and model, fall back to B" },
      { type: "route", model: "catwiring-mr04-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], expectIsFallback: true, expectPrimaryProviderRef: "self", label: "A fails; B serves as the router's fallback" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Model router 'catwiring-rtr-router-keeps-model-adds-fallbacks-r1-{{run_id}}' → provider=catwiring-rt-router-keeps-model-adds-fallbacks-{{run_id}}, model=catwiring-mr04-{{run_id}}"], waitSeconds: 2, label: "trail: the router kept the provider the caller named" },
      { type: "assertFallbackRow", providerRef: "b", fallbackIndex: 1, expectStatus: "success", waitSeconds: 0, label: "B served as fallback 1" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-empty-target",
    title: "A classifier model router with no target for a request follows empty_target",
    description: "A classifier router takes this run's own model and has no targets, so it has no target for any tier. With empty_target next_router it passes the request to the next router, a conditional one that sends it to B. With empty_target original_model, the default, it keeps the request and A, the provider the caller named, serves. The route is the same whether or not the classifier produces a tier, so the scenario does not depend on what the classifier answers. An unknown empty_target is refused.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["catwiring-mr05-{{run_id}}", MODEL_B], aliases: { "catwiring-mr05-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr05-{{run_id}}"], aliases: { "catwiring-mr05-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", ref: "r1", mode: "classifier", classifier: (nameOf) => ({ classifier: "llm", llm: { provider: nameOf("self"), model: MODEL_B }, keywords: { simple_keywords: ["what time is it"], medium_keywords: ["summarise this paragraph"], complex_keywords: ["design a distributed cache"] } }), cel: "model == 'catwiring-mr05-{{run_id}}'", priority: 0, emptyTarget: "next_router", label: "create a classifier router with no targets and empty_target next_router" },
      { type: "createRouter", ref: "r2", cel: "model == 'catwiring-mr05-{{run_id}}'", priority: 1, targets: [{ providerRef: "b", model: "catwiring-mr05-{{run_id}}" }], label: "create a conditional router after it: this run's model goes to B" },
      { type: "route", model: "catwiring-mr05-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], label: "next_router: the request reaches the second router and B serves" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Model router 'catwiring-rtr-router-empty-target-r1-{{run_id}}'", "; trying the next router", "Model router 'catwiring-rtr-router-empty-target-r2-{{run_id}}' → provider=catwiring-rt-router-empty-target-b-{{run_id}}"], waitSeconds: 2, label: "trail: the classifier router passes the request on" },
      { type: "routerApi", ref: "r1", method: "PUT", path: ["routing"], body: { empty_target: "original_model" }, expectStatus: 200, label: "set empty_target to original_model" },
      { type: "route", model: "catwiring-mr05-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, label: "original_model: the classifier router keeps the request and A serves" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["; keeping provider=catwiring-rt-router-empty-target-{{run_id}}, model=catwiring-mr05-{{run_id}}"], expectAbsent: ["Model router 'catwiring-rtr-router-empty-target-r2-{{run_id}}'"], waitSeconds: 2, label: "trail: the classifier router keeps the request and the second router is not asked" },
      { type: "routerApi", ref: "r1", method: "PUT", path: ["routing"], body: { empty_target: "skip" }, expectStatus: 400, expectErrorSubstr: "invalid empty_target \"skip\": must be one of: next_router, original_model", label: "an unknown empty_target is refused (400)" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-same-classifier-settings",
    title: "Classifier model routers with the same classifier settings classify a request once",
    description: "Two classifier routers take this run's own model, classify with the same settings and have no targets, so each passes the request to the next router; a conditional router after them sends it to B. The second router reuses the first one's answer: the trail holds one classification, whether it produced a tier or not, and each router's own line.",
    steps: [
      { type: "addProvider", ref: "self", keys: [key({ id: "ka", models: ["catwiring-mr06-{{run_id}}", MODEL_B], aliases: { "catwiring-mr06-{{run_id}}": MODEL_B } })] },
      { type: "addProvider", ref: "b", keys: [key({ id: "kb", models: ["catwiring-mr06-{{run_id}}"], aliases: { "catwiring-mr06-{{run_id}}": MODEL_B } })] },
      { type: "createRouter", ref: "r1", mode: "classifier", classifier: sameClassifierSettings, cel: "model == 'catwiring-mr06-{{run_id}}'", priority: 0, emptyTarget: "next_router", label: "create a classifier router with no targets and empty_target next_router" },
      { type: "createRouter", ref: "r2", mode: "classifier", classifier: sameClassifierSettings, cel: "model == 'catwiring-mr06-{{run_id}}'", priority: 1, emptyTarget: "next_router", label: "create a second classifier router with the same classifier settings" },
      { type: "createRouter", ref: "r3", cel: "model == 'catwiring-mr06-{{run_id}}'", priority: 2, targets: [{ providerRef: "b", model: "catwiring-mr06-{{run_id}}" }], label: "create a conditional router after them: this run's model goes to B" },
      { type: "route", model: "catwiring-mr06-{{run_id}}", useVk: false, requestId: true, expectStatus: 200, expectProviderOneOf: ["b"], label: "both classifier routers pass the request on and B serves" },
      { type: "assertRoutingTrail", fromRequest: true, expectSubstrings: ["Model router 'catwiring-rtr-router-same-classifier-settings-r1-{{run_id}}'", "Model router 'catwiring-rtr-router-same-classifier-settings-r2-{{run_id}}'", "; trying the next router", "Model router 'catwiring-rtr-router-same-classifier-settings-r3-{{run_id}}' → provider=catwiring-rt-router-same-classifier-settings-b-{{run_id}}"], expectCounts: [{ substring: "LLM complexity", count: 1 }], waitSeconds: 2, label: "trail: one classification, used by both classifier routers" },
      { type: "cleanup" },
    ],
  },
  {
    id: "router-api-refusals",
    title: "The model router API refuses a tier on a conditional router, a name taken in its scope and an unknown mode",
    description: "A conditional router's targets name no tier, so a target with one is refused. A name another router in the same scope already has is refused on create, since names are kept apart within a scope. An unknown mode is refused, and a conditional router has no classifier to reset. Each refusal is a 400 or a 409 with a message that says what to fix.",
    steps: [
      { type: "createRouter", ref: "r1", cel: "model == 'catwiring-mr06-{{run_id}}'", label: "create a conditional router" },
      { type: "routerApi", ref: "r1", method: "PUT", path: ["targets"], body: { targets: [{ tier: "SIMPLE", model: "catwiring-mr06-{{run_id}}", weight: 1 }] }, expectStatus: 400, expectErrorSubstr: "a conditional router's targets take no tier", label: "a tier on a conditional router's target is refused (400)" },
      { type: "routerApi", method: "POST", body: ({ routerId }) => ({ id: "catwiring-rtr-router-api-refusals-dup-{{run_id}}", name: routerId("r1"), mode: "conditional" }), expectStatus: 409, expectErrorSubstr: "a model router named \"catwiring-rtr-router-api-refusals-r1-{{run_id}}\" already exists in this scope", label: "a name another global router has is refused (409)" },
      { type: "routerApi", method: "POST", body: { id: "catwiring-rtr-router-api-refusals-mode-{{run_id}}", name: "catwiring-rtr-router-api-refusals-mode-{{run_id}}", mode: "weighted" }, expectStatus: 400, expectErrorSubstr: "invalid mode \"weighted\": must be one of: classifier, conditional", label: "an unknown mode is refused (400)" },
      { type: "routerApi", ref: "r1", method: "POST", path: ["reset"], expectStatus: 400, expectErrorSubstr: "is a conditional router and has no classifier to reset", label: "a conditional router has no classifier to reset (400)" },
      { type: "cleanup" },
    ],
  },
];

const collection = buildCollection({
  id: "bifrost-routing-wiring",
  name: "Bifrost Routing Wiring (Governance x Catalog)",
  description:
    "Governance × model-catalog routing wiring. Each scenario stands up an isolated, run-namespaced " +
    "custom provider backed by real OpenAI, gates it with keys and a virtual key, then drives inference " +
    "and asserts the route via extra_fields.routing_info (success), the error message (rejection), and " +
    "the stored log. Machine-generated by runners/build-routing-wiring.mjs — do not hand-edit.",
  expandedScenarios: SCENARIOS.map(expandScenario),
  // OSS locks /api behind the setup token until an admin exists; the e2e profile sets one.
  setupToken: true,
});

writeCollection(resolveOutPath(DEFAULT_OUT), collection, SCENARIOS.map((s) => s.id));
