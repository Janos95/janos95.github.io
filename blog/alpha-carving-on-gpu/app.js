const DRAWINGS = [
  { file: "drawing_06.txt", name: "Scattered segment field" },
  { file: "drawing.txt", name: "Open hooked channel" },
  { file: "drawing_01.txt", name: "Nested spiral loops" },
  { file: "drawing_03.txt", name: "Closed pocket with notch" },
  { file: "drawing_04.txt", name: "Crossing diagonal bars" },
  { file: "drawing_05.txt", name: "Bent polyline" },
];

const INVALID = 0xffffffff;
const WORKGROUP_2D = 16;
const WORKGROUP_1D = 256;
const PARAM_BYTES = 48;

const els = {
  canvas: document.getElementById("gpuCanvas"),
  overlay: document.getElementById("editOverlay"),
  unsupported: document.getElementById("unsupported"),
  meshSelect: document.getElementById("meshSelect"),
  resolutionSelect: document.getElementById("resolutionSelect"),
  alphaSlider: document.getElementById("alphaSlider"),
  alphaValue: document.getElementById("alphaValue"),
  clearCustom: document.getElementById("clearCustom"),
  gpuText: document.getElementById("gpuText"),
  segmentText: document.getElementById("segmentText"),
};

for (const drawing of DRAWINGS) {
  const option = document.createElement("option");
  option.value = drawing.file;
  option.textContent = drawing.name;
  els.meshSelect.append(option);
}

function parseDrawing(text) {
  const verts = [];
  const segments = [];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const parts = line.split(/\s+/);
    if (parts[0] === "v" && parts.length >= 3) {
      verts.push([Number(parts[1]), Number(parts[2])]);
    } else if (parts[0] === "l" && parts.length >= 3) {
      const a = verts[Number(parts[1]) - 1];
      const b = verts[Number(parts[2]) - 1];
      if (a && b) segments.push([a[0], a[1], b[0], b[1]]);
    }
  }

  if (!segments.length) {
    throw new Error("Drawing has no line segments.");
  }

  return { verts, segments };
}

function makeParamsBufferData({ width, height, segmentCount, passStride, alpha, showCenters }) {
  const data = new ArrayBuffer(PARAM_BYTES);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);

  u32[0] = width;
  u32[1] = height;
  u32[2] = segmentCount;
  u32[3] = passStride;
  f32[4] = alpha;
  f32[5] = 2 / width;
  f32[6] = 2 / height;
  f32[7] = Math.max(0.0035, 2.2 / width);
  u32[8] = showCenters ? 1 : 0;
  u32[9] = 0;
  u32[10] = 0;
  u32[11] = 0;

  return data;
}

function nextPowerOfTwo(value) {
  let x = 1;
  while (x < value) x <<= 1;
  return x;
}

function makeJfaSteps(width, height) {
  const maxDim = Math.max(width, height);
  const steps = [];
  for (let step = nextPowerOfTwo(maxDim) >> 1; step >= 1; step >>= 1) {
    steps.push(step);
  }
  return steps;
}

function dispatch2D(pass, width, height) {
  pass.dispatchWorkgroups(
    Math.ceil(width / WORKGROUP_2D),
    Math.ceil(height / WORKGROUP_2D),
  );
}

class AlphaCarver {
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.alpha = Number(els.alphaSlider.value);
    this.width = 1024;
    this.height = 1024;
    this.mesh = null;
    this.resources = null;
    this.busy = false;
    this.dirty = true;
    this.renderDirty = true;
    this.finalSeedIsA = true;
    this.lastGpuMs = 0;

    this.createPipelines();
  }

  createPipelines() {
    const distanceModule = this.device.createShaderModule({
      label: "distance shader",
      code: `
struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> clearance: array<f32>;

fn cell_world(x: u32, y: u32) -> vec2<f32> {
  return vec2<f32>(
    -1.0 + (f32(x) + 0.5) * params.cellSizeX,
    -1.0 + (f32(y) + 0.5) * params.cellSizeY
  );
}

fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let denom = max(dot(ab, ab), 1e-12);
  let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
  return length(p - (a + t * ab));
}

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  let p = cell_world(gid.x, gid.y);
  var best = 1e9;

  for (var i = 0u; i < params.segmentCount; i = i + 1u) {
    let s = segments[i];
    best = min(best, point_segment_distance(p, s.xy, s.zw));
  }

  clearance[idx] = best;
}
`,
    });

    const initModule = this.device.createShaderModule({
      label: "init shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> clearance: array<f32>;
@group(0) @binding(2) var<storage, read_write> valid: array<u32>;
@group(0) @binding(3) var<storage, read_write> parent: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u && gid.y == 0u) {
    atomicStore(&parent[0], 0u);
  }

  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  let id = idx + 1u;
  let ok = clearance[idx] >= params.alpha;
  valid[idx] = select(0u, 1u, ok);
  atomicStore(&parent[id], select(INVALID, id, ok));
}
`,
    });

    const unionModule = this.device.createShaderModule({
      label: "union shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read_write> parent: array<atomic<u32>>;

fn find_root(start: u32) -> u32 {
  var x = start;

  loop {
    let p = atomicLoad(&parent[x]);
    if (p == x) {
      return x;
    }

    let gp = atomicLoad(&parent[p]);
    if (gp != p) {
      _ = atomicCompareExchangeWeak(&parent[x], p, gp);
    }
    x = gp;
  }
}

fn unite(a: u32, b: u32) {
  var x = a;
  var y = b;

  loop {
    let rx = find_root(x);
    let ry = find_root(y);

    if (rx == ry) {
      return;
    }

    let hi = max(rx, ry);
    let lo = min(rx, ry);
    let result = atomicCompareExchangeWeak(&parent[hi], hi, lo);
    if (result.exchanged) {
      return;
    }

    x = hi;
    y = lo;
  }
}

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  if (valid[idx] == 0u) {
    return;
  }

  let id = idx + 1u;

  if (gid.x == 0u || gid.y == 0u || gid.x + 1u == params.width || gid.y + 1u == params.height) {
    unite(id, 0u);
  }

  if (gid.x + 1u < params.width && valid[idx + 1u] != 0u) {
    unite(id, id + 1u);
  }

  if (gid.y + 1u < params.height && valid[idx + params.width] != 0u) {
    unite(id, id + params.width);
  }
}
`,
    });

    const compressModule = this.device.createShaderModule({
      label: "compress shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> parent: array<atomic<u32>>;

@compute @workgroup_size(${WORKGROUP_1D}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.width * params.height + 1u;
  let id = gid.x;
  if (id >= total) {
    return;
  }

  let p = atomicLoad(&parent[id]);
  if (p == INVALID || p == id) {
    return;
  }

  let gp = atomicLoad(&parent[p]);
  if (gp != INVALID && gp != p) {
    atomicStore(&parent[id], gp);
  }
}
`,
    });

    const seedModule = this.device.createShaderModule({
      label: "seed shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> seeds: array<u32>;

fn find_root(start: u32) -> u32 {
  var x = start;

  loop {
    let p = atomicLoad(&parent[x]);
    if (p == x) {
      return x;
    }

    let gp = atomicLoad(&parent[p]);
    if (gp != p) {
      _ = atomicCompareExchangeWeak(&parent[x], p, gp);
    }
    x = gp;
  }
}

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  if (valid[idx] == 0u) {
    seeds[idx] = INVALID;
    return;
  }

  let root = find_root(idx + 1u);
  seeds[idx] = select(INVALID, idx, root == 0u);
}
`,
    });

    const jfaModule = this.device.createShaderModule({
      label: "jfa shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seedIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> seedOut: array<u32>;

fn seed_world(seed: u32) -> vec2<f32> {
  let sx = seed % params.width;
  let sy = seed / params.width;
  return vec2<f32>(
    -1.0 + (f32(sx) + 0.5) * params.cellSizeX,
    -1.0 + (f32(sy) + 0.5) * params.cellSizeY
  );
}

fn cell_world(x: u32, y: u32) -> vec2<f32> {
  return vec2<f32>(
    -1.0 + (f32(x) + 0.5) * params.cellSizeX,
    -1.0 + (f32(y) + 0.5) * params.cellSizeY
  );
}

fn consider(candidate: u32, p: vec2<f32>, bestSeed: ptr<function, u32>, bestD2: ptr<function, f32>) {
  if (candidate == INVALID) {
    return;
  }
  let sp = seed_world(candidate);
  let d = sp - p;
  let d2 = dot(d, d);
  if (d2 < *bestD2) {
    *bestD2 = d2;
    *bestSeed = candidate;
  }
}

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  let p = cell_world(gid.x, gid.y);
  var bestSeed = seedIn[idx];
  var bestD2 = 1e30;
  consider(bestSeed, p, &bestSeed, &bestD2);

  let stride = i32(params.passStride);
  let baseX = i32(gid.x);
  let baseY = i32(gid.y);

  for (var oy = -1; oy <= 1; oy = oy + 1) {
    for (var ox = -1; ox <= 1; ox = ox + 1) {
      let nx = baseX + ox * stride;
      let ny = baseY + oy * stride;
      if (nx >= 0 && ny >= 0 && nx < i32(params.width) && ny < i32(params.height)) {
        let nidx = u32(ny) * params.width + u32(nx);
        consider(seedIn[nidx], p, &bestSeed, &bestD2);
      }
    }
  }

  seedOut[idx] = bestSeed;
}
`,
    });

    const composeModule = this.device.createShaderModule({
      label: "compose shader",
      code: `
const INVALID: u32 = 0xffffffffu;

struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;

fn cell_world(x: u32, y: u32) -> vec2<f32> {
  return vec2<f32>(
    -1.0 + (f32(x) + 0.5) * params.cellSizeX,
    -1.0 + (f32(y) + 0.5) * params.cellSizeY
  );
}

fn seed_world(seed: u32) -> vec2<f32> {
  let sx = seed % params.width;
  let sy = seed / params.width;
  return cell_world(sx, sy);
}

@compute @workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let idx = gid.y * params.width + gid.x;
  let p = cell_world(gid.x, gid.y);
  let seed = seeds[idx];
  var carved = false;
  var distToReachable = 1e9;

  if (seed != INVALID) {
    let d = p - seed_world(seed);
    distToReachable = length(d);
    carved = distToReachable <= params.alpha + 0.5 * max(params.cellSizeX, params.cellSizeY);
  }

  let color = select(vec3<f32>(1.0, 0.79, 0.76), vec3<f32>(0.78, 0.91, 1.0), carved);

  textureStore(outputTex, vec2<i32>(i32(gid.x), i32(params.height - 1u - gid.y)), vec4<f32>(color, 1.0));
}
`,
    });

    const renderModule = this.device.createShaderModule({
      label: "render shader",
      code: `
struct Params {
  width: u32,
  height: u32,
  segmentCount: u32,
  passStride: u32,
  alpha: f32,
  cellSizeX: f32,
  cellSizeY: f32,
  lineWidth: f32,
  showCenters: u32,
  colorMode: u32,
  pad0: u32,
  pad1: u32,
}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var texSampler: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> segments: array<vec4<f32>>;

fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let denom = max(dot(ab, ab), 1e-12);
  let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
  return length(p - (a + t * ab));
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );

  let pos = positions[vertexIndex];
  var out: VSOut;
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + vec2<f32>(0.5);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  var color = textureSampleLevel(tex, texSampler, in.uv, 0.0).rgb;
  let p = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);

  var meshDistance = 1e9;
  for (var i = 0u; i < params.segmentCount; i = i + 1u) {
    let s = segments[i];
    meshDistance = min(meshDistance, point_segment_distance(p, s.xy, s.zw));
  }

  let meshInk = 1.0 - smoothstep(params.lineWidth, params.lineWidth * 2.25, meshDistance);
  color = color * (1.0 - meshInk) + vec3<f32>(0.02, 0.025, 0.02) * meshInk;
  return vec4<f32>(color, 1.0);
}
`,
    });

    this.distancePipeline = this.device.createComputePipeline({
      label: "distance pipeline",
      layout: "auto",
      compute: { module: distanceModule },
    });
    this.initPipeline = this.device.createComputePipeline({
      label: "init pipeline",
      layout: "auto",
      compute: { module: initModule },
    });
    this.unionPipeline = this.device.createComputePipeline({
      label: "union pipeline",
      layout: "auto",
      compute: { module: unionModule },
    });
    this.compressPipeline = this.device.createComputePipeline({
      label: "compress pipeline",
      layout: "auto",
      compute: { module: compressModule },
    });
    this.seedPipeline = this.device.createComputePipeline({
      label: "seed pipeline",
      layout: "auto",
      compute: { module: seedModule },
    });
    this.jfaPipeline = this.device.createComputePipeline({
      label: "jfa pipeline",
      layout: "auto",
      compute: { module: jfaModule },
    });
    this.composePipeline = this.device.createComputePipeline({
      label: "compose pipeline",
      layout: "auto",
      compute: { module: composeModule },
    });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "render pipeline",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs" },
      fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  setAlpha(alpha) {
    this.alpha = alpha;
    this.dirty = true;
  }

  setResolution(size) {
    this.width = size;
    this.height = size;
    this.createResources();
    this.computeDistance();
    this.dirty = true;
    this.renderDirty = true;
  }

  setMesh(mesh) {
    this.mesh = mesh;
    if (!this.resources) {
      this.createResources();
    } else {
      this.createSegmentBuffer();
      this.createBindGroups();
    }
    this.computeDistance();
    this.dirty = true;
    this.renderDirty = true;
  }

  createResources() {
    const device = this.device;
    const width = this.width;
    const height = this.height;
    const cells = width * height;
    const parentCells = cells + 1;
    const steps = makeJfaSteps(width, height);

    this.resources?.outputTexture?.destroy();

    this.resources = {
      cells,
      parentCells,
      steps,
      paramBase: device.createBuffer({
        label: "params base",
        size: PARAM_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      jfaParams: steps.map((step) =>
        device.createBuffer({
          label: `params jfa ${step}`,
          size: PARAM_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      ),
      clearance: device.createBuffer({
        label: "clearance",
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      valid: device.createBuffer({
        label: "valid centers",
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      parent: device.createBuffer({
        label: "union find parent",
        size: parentCells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      seedA: device.createBuffer({
        label: "jfa seed A",
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      seedB: device.createBuffer({
        label: "jfa seed B",
        size: cells * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      outputTexture: device.createTexture({
        label: "composed carve texture",
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      }),
      sampler: device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      }),
    };

    this.createSegmentBuffer();
    this.createBindGroups();
    this.updateParamBuffers();
  }

  createSegmentBuffer() {
    if (!this.mesh || !this.resources) return;

    const segments = new Float32Array(Math.max(4, this.mesh.segments.length * 4));
    for (let i = 0; i < this.mesh.segments.length; i += 1) {
      segments.set(this.mesh.segments[i], i * 4);
    }

    this.resources.segmentBuffer = this.device.createBuffer({
      label: "segments",
      size: segments.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.resources.segmentBuffer, 0, segments);
  }

  updateParamBuffers() {
    if (!this.resources || !this.mesh) return;

    const base = {
      width: this.width,
      height: this.height,
      segmentCount: this.mesh.segments.length,
      alpha: this.alpha,
      showCenters: false,
    };

    this.device.queue.writeBuffer(
      this.resources.paramBase,
      0,
      makeParamsBufferData({ ...base, passStride: 0 }),
    );

    for (let i = 0; i < this.resources.steps.length; i += 1) {
      this.device.queue.writeBuffer(
        this.resources.jfaParams[i],
        0,
        makeParamsBufferData({ ...base, passStride: this.resources.steps[i] }),
      );
    }
  }

  createBindGroups() {
    const r = this.resources;
    if (!r || !r.segmentBuffer) return;

    this.distanceBindGroup = this.device.createBindGroup({
      label: "distance bind group",
      layout: this.distancePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.segmentBuffer } },
        { binding: 2, resource: { buffer: r.clearance } },
      ],
    });

    this.initBindGroup = this.device.createBindGroup({
      label: "init bind group",
      layout: this.initPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.clearance } },
        { binding: 2, resource: { buffer: r.valid } },
        { binding: 3, resource: { buffer: r.parent } },
      ],
    });

    this.unionBindGroup = this.device.createBindGroup({
      label: "union bind group",
      layout: this.unionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.valid } },
        { binding: 2, resource: { buffer: r.parent } },
      ],
    });

    this.compressBindGroup = this.device.createBindGroup({
      label: "compress bind group",
      layout: this.compressPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.parent } },
      ],
    });

    this.seedBindGroup = this.device.createBindGroup({
      label: "seed bind group",
      layout: this.seedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.valid } },
        { binding: 2, resource: { buffer: r.parent } },
        { binding: 3, resource: { buffer: r.seedA } },
      ],
    });

    this.jfaBindGroupsAtoB = r.steps.map((_, i) =>
      this.device.createBindGroup({
        label: `jfa A to B ${i}`,
        layout: this.jfaPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: r.jfaParams[i] } },
          { binding: 1, resource: { buffer: r.seedA } },
          { binding: 2, resource: { buffer: r.seedB } },
        ],
      }),
    );

    this.jfaBindGroupsBtoA = r.steps.map((_, i) =>
      this.device.createBindGroup({
        label: `jfa B to A ${i}`,
        layout: this.jfaPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: r.jfaParams[i] } },
          { binding: 1, resource: { buffer: r.seedB } },
          { binding: 2, resource: { buffer: r.seedA } },
        ],
      }),
    );

    this.composeBindGroupA = this.device.createBindGroup({
      label: "compose bind group A",
      layout: this.composePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.seedA } },
        { binding: 2, resource: r.outputTexture.createView() },
      ],
    });

    this.composeBindGroupB = this.device.createBindGroup({
      label: "compose bind group B",
      layout: this.composePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: r.paramBase } },
        { binding: 1, resource: { buffer: r.seedB } },
        { binding: 2, resource: r.outputTexture.createView() },
      ],
    });

    this.renderBindGroup = this.device.createBindGroup({
      label: "render bind group",
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: r.sampler },
        { binding: 1, resource: r.outputTexture.createView() },
        { binding: 2, resource: { buffer: r.paramBase } },
        { binding: 3, resource: { buffer: r.segmentBuffer } },
      ],
    });
  }

  computeDistance() {
    if (!this.resources || !this.mesh) return;

    this.updateParamBuffers();
    const encoder = this.device.createCommandEncoder({ label: "distance encoder" });
    const pass = encoder.beginComputePass({ label: "distance pass" });
    pass.setPipeline(this.distancePipeline);
    pass.setBindGroup(0, this.distanceBindGroup);
    dispatch2D(pass, this.width, this.height);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  encodeAlphaUpdate(encoder) {
    const r = this.resources;
    this.updateParamBuffers();
    const cellSize = 2 / this.width;
    const radiusCells = Math.max(1, Math.ceil(this.alpha / cellSize) + 1);
    const startStride = Math.min(nextPowerOfTwo(radiusCells), r.steps[0]);
    const firstJfaPass = Math.max(0, r.steps.findIndex((step) => step <= startStride));

    let pass = encoder.beginComputePass({ label: "init alpha pass" });
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.initBindGroup);
    dispatch2D(pass, this.width, this.height);
    pass.end();

    pass = encoder.beginComputePass({ label: "union alpha pass" });
    pass.setPipeline(this.unionPipeline);
    pass.setBindGroup(0, this.unionBindGroup);
    dispatch2D(pass, this.width, this.height);
    pass.end();

    const compressionPasses = 2;
    for (let i = 0; i < compressionPasses; i += 1) {
      pass = encoder.beginComputePass({ label: `compress pass ${i + 1}` });
      pass.setPipeline(this.compressPipeline);
      pass.setBindGroup(0, this.compressBindGroup);
      pass.dispatchWorkgroups(Math.ceil(r.parentCells / WORKGROUP_1D));
      pass.end();
    }

    pass = encoder.beginComputePass({ label: "reachable seed pass" });
    pass.setPipeline(this.seedPipeline);
    pass.setBindGroup(0, this.seedBindGroup);
    dispatch2D(pass, this.width, this.height);
    pass.end();

    let readIsA = true;
    for (let i = firstJfaPass; i < r.steps.length; i += 1) {
      pass = encoder.beginComputePass({ label: `jfa pass ${r.steps[i]}` });
      pass.setPipeline(this.jfaPipeline);
      pass.setBindGroup(0, readIsA ? this.jfaBindGroupsAtoB[i] : this.jfaBindGroupsBtoA[i]);
      dispatch2D(pass, this.width, this.height);
      pass.end();
      readIsA = !readIsA;
    }
    this.finalSeedIsA = readIsA;

    pass = encoder.beginComputePass({ label: "compose pass" });
    pass.setPipeline(this.composePipeline);
    pass.setBindGroup(0, this.finalSeedIsA ? this.composeBindGroupA : this.composeBindGroupB);
    dispatch2D(pass, this.width, this.height);
    pass.end();
  }

  updateAlphaOnGpu() {
    if (!this.resources || !this.mesh || this.busy) return;

    try {
      this.dirty = false;
      this.busy = true;
      const started = performance.now();
      const encoder = this.device.createCommandEncoder({ label: "alpha update encoder" });
      this.encodeAlphaUpdate(encoder);
      this.encodeRender(encoder);
      this.device.queue.submit([encoder.finish()]);
      this.renderDirty = false;

      this.device.queue.onSubmittedWorkDone().then(() => {
        this.lastGpuMs = performance.now() - started;
        this.busy = false;
        const updateFps = 1000 / Math.max(this.lastGpuMs, 0.001);
        els.gpuText.textContent = `${this.lastGpuMs.toFixed(1)} ms / ${updateFps.toFixed(0)} fps`;
        if (this.dirty) {
          this.updateAlphaOnGpu();
        }
      }).catch((error) => {
        this.busy = false;
        console.error(error);
      });
    } catch (error) {
      this.busy = false;
      console.error(error);
    }
  }

  render() {
    if (!this.resources || !this.renderBindGroup) return;

    const encoder = this.device.createCommandEncoder({ label: "render encoder" });
    this.encodeRender(encoder);
    this.device.queue.submit([encoder.finish()]);
    this.renderDirty = false;
  }

  encodeRender(encoder) {
    if (!this.resources || !this.renderBindGroup) return;

    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      label: "render pass",
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.14, g: 0.15, b: 0.13, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.draw(3);
    pass.end();
  }
}

function resizeCanvas(context, device, format) {
  const rect = els.canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.max(320, Math.floor(Math.min(rect.width, rect.height) * dpr));

  if (els.canvas.width !== size || els.canvas.height !== size) {
    els.canvas.width = size;
    els.canvas.height = size;
    context.configure({
      device,
      format,
      alphaMode: "opaque",
    });
    return true;
  }
  return false;
}

function resizeEditOverlay() {
  if (!els.overlay) return false;

  const rect = els.canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));

  if (els.overlay.width !== width || els.overlay.height !== height) {
    els.overlay.width = width;
    els.overlay.height = height;
    return true;
  }
  return false;
}

function screenToWorld(event) {
  const rect = els.canvas.getBoundingClientRect();
  const u = (event.clientX - rect.left) / rect.width;
  const v = (event.clientY - rect.top) / rect.height;

  return {
    x: Math.max(-1, Math.min(1, 2 * u - 1)),
    y: Math.max(-1, Math.min(1, 2 * v - 1)),
  };
}

function worldToScreen(point) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: ((point.x + 1) * 0.5) * rect.width,
    y: ((point.y + 1) * 0.5) * rect.height,
  };
}

async function loadDrawing(file) {
  const response = await fetch(`./${file}?t=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Could not load ${file}: ${response.status}`);
  }
  return parseDrawing(await response.text());
}

async function main() {
  window.addEventListener("error", (event) => {
    console.error(event.message || "Runtime error");
  });
  window.addEventListener("unhandledrejection", (event) => {
    console.error(event.reason);
  });

  if (!navigator.gpu) {
    els.unsupported.hidden = false;
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    els.unsupported.hidden = false;
    return;
  }

  const device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", (event) => {
    console.error(event.error);
  });
  const context = els.canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  resizeCanvas(context, device, format);

  const carver = new AlphaCarver(device, context, format);
  window.carver = carver;

  let currentMesh = null;
  const customState = {
    using: false,
    pending: null,
    hover: null,
    segments: [],
  };

  function cloneSegments(segments) {
    return segments.map((segment) => [...segment]);
  }

  function customVertices() {
    const vertices = [];
    for (const segment of customState.segments) {
      vertices.push([segment[0], segment[1]], [segment[2], segment[3]]);
    }
    if (customState.pending) {
      vertices.push([customState.pending.x, customState.pending.y]);
    }
    return vertices;
  }

  function customMesh() {
    return {
      verts: customVertices(),
      segments: customState.segments.map((segment) => [...segment]),
    };
  }

  function updateCustomButtons() {
    const hasSegments = currentMesh && currentMesh.segments.length > 0;
    els.clearCustom.disabled = !customState.pending && !hasSegments;
  }

  function drawOverlay() {
    resizeEditOverlay();
    const overlay = els.overlay;
    const ctx = overlay.getContext("2d");
    const rect = els.canvas.getBoundingClientRect();
    const scale = overlay.width / Math.max(rect.width, 1);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!customState.pending) {
      return;
    }

    ctx.setTransform(scale, 0, 0, scale, 0, 0);

    const pending = worldToScreen(customState.pending);
    ctx.fillStyle = "#2f6fba";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pending.x, pending.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (customState.hover) {
      const hover = worldToScreen(customState.hover);
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "#2f6fba";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pending.x, pending.y);
      ctx.lineTo(hover.x, hover.y);
      ctx.stroke();
      ctx.restore();
    }
  }

  function updateCustomMesh() {
    customState.using = true;
    const mesh = customMesh();
    currentMesh = mesh;
    carver.setMesh(mesh);
    els.segmentText.textContent = String(customState.segments.length);
    updateCustomButtons();
    drawOverlay();
  }

  function beginCustomDrawing() {
    if (!customState.using) {
      customState.segments = currentMesh ? cloneSegments(currentMesh.segments) : [];
      customState.using = true;
    }
  }

  async function applyMeshSelection() {
    const file = els.meshSelect.value;
    if (!file) {
      return;
    }

    customState.pending = null;
    customState.hover = null;
    customState.using = false;
    customState.segments = [];
    const mesh = await loadDrawing(file);
    currentMesh = mesh;
    carver.setMesh(mesh);
    els.segmentText.textContent = String(mesh.segments.length);
    updateCustomButtons();
    drawOverlay();
  }

  els.alphaSlider.addEventListener("input", () => {
    const alpha = Number(els.alphaSlider.value);
    els.alphaValue.value = alpha.toFixed(3);
    carver.setAlpha(alpha);
  });

  els.meshSelect.addEventListener("change", () => {
    applyMeshSelection().catch((error) => {
      console.error(error);
    });
  });

  els.canvas.addEventListener("click", (event) => {
    beginCustomDrawing();

    const point = screenToWorld(event);
    if (!customState.pending) {
      customState.pending = point;
      updateCustomButtons();
      drawOverlay();
      return;
    }

    const start = customState.pending;
    if (Math.hypot(point.x - start.x, point.y - start.y) > 1e-4) {
      customState.segments.push([start.x, start.y, point.x, point.y]);
    }
    customState.pending = null;
    customState.hover = null;
    updateCustomMesh();
  });

  els.canvas.addEventListener("mousemove", (event) => {
    if (!customState.pending) return;
    customState.hover = screenToWorld(event);
    drawOverlay();
  });

  els.canvas.addEventListener("mouseleave", () => {
    customState.hover = null;
    drawOverlay();
  });

  els.clearCustom.addEventListener("click", () => {
    customState.pending = null;
    customState.hover = null;
    customState.segments = [];
    updateCustomMesh();
  });

  els.resolutionSelect.addEventListener("change", () => {
    const size = Number(els.resolutionSelect.value);
    carver.setResolution(size);
  });

  window.addEventListener("resize", () => {
    resizeCanvas(context, device, format);
    drawOverlay();
  });

  device.lost.then((info) => {
    console.error(`GPU device lost: ${info.message || info.reason}`);
  });

  function frame() {
    try {
      if (resizeCanvas(context, device, format)) {
        carver.renderDirty = true;
        drawOverlay();
      } else if (resizeEditOverlay()) {
        drawOverlay();
      }
      if (carver.dirty && !carver.busy) {
        carver.updateAlphaOnGpu();
      } else if (carver.renderDirty && !carver.busy) {
        carver.render();
      }
    } catch (error) {
      console.error(error);
    }

    requestAnimationFrame(frame);
  }

  els.alphaValue.value = Number(els.alphaSlider.value).toFixed(3);
  await applyMeshSelection();
  requestAnimationFrame(frame);
}

main().catch((error) => {
  console.error(error);
});
