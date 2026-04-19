(function (global) {
  'use strict';

  var data = global.FractalWalkthroughData;

  if (!data) {
    return;
  }

  global.addEventListener('DOMContentLoaded', function () {
    var canvas = document.getElementById('walkthroughCanvas');
    var overlay = document.getElementById('walkthroughOverlay');
    var nextButton = document.getElementById('walkthroughNext');
    var resetButton = document.getElementById('walkthroughReset');
    var caption = document.getElementById('walkthroughCaption');
    var stepStat = document.getElementById('walkthroughStep');
    var ctx;
    var imageData;
    var offscreen;
    var offscreenCtx;
    var patchCanvas;
    var patchCtx;
    var lennaPixels = null;
    var randomPixels = null;
    var fullMapping = null;
    var iterationFrames = [];
    var branchNodes = [];
    var stageIndex = 0;
    var ready = false;
    var isAnimating = false;
    var animationTimer = 0;
    var iterationDisplay = '';

    var SIZE = data.size;
    var RANGE_SIZE = data.rangeSize;
    var DOMAIN_SIZE = data.domainSize;
    var RANGE_AREA = RANGE_SIZE * RANGE_SIZE;
    var DOMAIN_AREA = DOMAIN_SIZE * DOMAIN_SIZE;
    var COLOR_CHANNELS = 3;
    var RANGE_BLOCKS_PER_SIDE = SIZE / RANGE_SIZE;
    var DOMAIN_BLOCKS_PER_SIDE = SIZE / DOMAIN_SIZE;
    var RANGE_COLOR = '#f97316';
    var DOMAIN_COLOR = '#2563eb';
    var ITERATION_COUNT = 6;
    var STEP6_TIME_SCALE = 1.5;
    var ANIMATION_TICK_MS = 33;
    var NODE_ANIMATION_MS = Math.round(520 * STEP6_TIME_SCALE);
    var NODE_GAP_MS = Math.round(60 * STEP6_TIME_SCALE);
    var FINAL_REVEAL_MS = Math.round(220 * STEP6_TIME_SCALE);
    var ITERATION_GAP_MS = Math.round(120 * STEP6_TIME_SCALE);
    var STAGES = [
      {
        frame: 'lenna',
        rootOnly: true,
        showDomain: false,
        visibleDepth: 0,
        caption: 'Start with one ' + RANGE_SIZE + ' x ' + RANGE_SIZE + ' range block in Lenna.'
      },
      {
        frame: 'lenna',
        rootOnly: true,
        showDomain: true,
        visibleDepth: 0,
        caption: 'The stored map picks one best ' + DOMAIN_SIZE + ' x ' + DOMAIN_SIZE + ' domain block and contracts it onto that range.'
      },
      {
        frame: 'lenna',
        visibleDepth: 1,
        caption: 'Inside that domain, one contained ' + RANGE_SIZE + ' x ' + RANGE_SIZE + ' range block is highlighted together with its own best ' + DOMAIN_SIZE + ' x ' + DOMAIN_SIZE + ' domain block.'
      },
      {
        frame: 'lenna',
        visibleDepth: 2,
        caption: 'One more click follows that child block one level deeper, adding another range-to-domain dependency.'
      },
      {
        frame: 'random',
        visibleDepth: 2,
        caption: 'Now replace Lenna with a random seed image while keeping the same highlighted dependency chain.'
      },
      {
        frame: 'iter',
        visibleDepth: 2,
        caption: 'Each iteration first animates the highlighted leaf-to-root block updates, then the full image advances. Repeating that several times shows the contraction toward the fixed point.'
      }
    ];

    if (!canvas || !overlay || !nextButton || !resetButton || !caption || !stepStat) {
      return;
    }

    ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    imageData = ctx.createImageData(SIZE, SIZE);
    offscreen = document.createElement('canvas');
    offscreen.width = SIZE;
    offscreen.height = SIZE;
    offscreenCtx = offscreen.getContext('2d');
    patchCanvas = document.createElement('canvas');
    patchCtx = patchCanvas.getContext('2d');
    if (!offscreenCtx || !patchCtx) {
      return;
    }

    function createSeededRandom(seed) {
      var state = (seed >>> 0) || 1;

      return function () {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 0xffffffff;
      };
    }

    function hashString(value) {
      var hash = 0;
      var i;

      for (i = 0; i < value.length; i++) {
        hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
      }

      return hash >>> 0;
    }

    function clamp255(value) {
      if (value < 0) {
        return 0;
      }
      if (value > 255) {
        return 255;
      }
      return value;
    }

    function toRgba(hex, alpha) {
      var clean = hex.replace('#', '');
      var r = parseInt(clean.slice(0, 2), 16);
      var g = parseInt(clean.slice(2, 4), 16);
      var b = parseInt(clean.slice(4, 6), 16);

      return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
    }

    function smoothstep(edge0, edge1, value) {
      var t;

      if (edge0 === edge1) {
        return value < edge0 ? 0 : 1;
      }

      t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)));
      return t * t * (3 - 2 * t);
    }

    function rangeCoords(rangeIndex) {
      return {
        x: (rangeIndex % RANGE_BLOCKS_PER_SIDE) * RANGE_SIZE,
        y: Math.floor(rangeIndex / RANGE_BLOCKS_PER_SIDE) * RANGE_SIZE
      };
    }

    function domainCoords(domainIndex) {
      return {
        x: (domainIndex % DOMAIN_BLOCKS_PER_SIDE) * DOMAIN_SIZE,
        y: Math.floor(domainIndex / DOMAIN_BLOCKS_PER_SIDE) * DOMAIN_SIZE
      };
    }

    function blockCenter(box) {
      return {
        x: box.x + box.size * 0.5,
        y: box.y + box.size * 0.5
      };
    }

    function getRangeBox(rangeIndex) {
      var coords = rangeCoords(rangeIndex);

      return {
        x: coords.x,
        y: coords.y,
        size: RANGE_SIZE
      };
    }

    function getDomainBox(domainIndex) {
      var coords = domainCoords(domainIndex);

      return {
        x: coords.x,
        y: coords.y,
        size: DOMAIN_SIZE
      };
    }

    function transformIndex(x, y, sym, size) {
      var rot = sym & 3;
      var flip = sym >= 4;
      var fx = flip ? size - 1 - x : x;
      var srcX;
      var srcY;

      switch (rot) {
        case 0:
          srcY = y;
          srcX = fx;
          break;
        case 1:
          srcY = fx;
          srcX = size - 1 - y;
          break;
        case 2:
          srcY = size - 1 - y;
          srcX = size - 1 - fx;
          break;
        case 3:
          srcY = size - 1 - fx;
          srcX = y;
          break;
        default:
          srcY = y;
          srcX = fx;
      }

      return srcY * size + srcX;
    }

    function transformBlock(src, srcOffset, sym, size, dest, destOffset) {
      var x;
      var y;
      var rowOffset;
      var srcIdx;

      for (y = 0; y < size; y++) {
        rowOffset = destOffset + y * size;
        for (x = 0; x < size; x++) {
          srcIdx = transformIndex(x, y, sym, size);
          dest[rowOffset + x] = src[srcOffset + srcIdx];
        }
      }
    }

    function downscaleBlock(src, srcOffset, size, dest, destOffset) {
      var half = size >> 1;
      var x;
      var y;
      var baseSrcY;
      var baseDest;
      var baseSrc;
      var a;
      var b;
      var c;
      var d;

      for (y = 0; y < half; y++) {
        baseSrcY = (y << 1) * size + srcOffset;
        baseDest = destOffset + y * half;
        for (x = 0; x < half; x++) {
          baseSrc = baseSrcY + (x << 1);
          a = src[baseSrc];
          b = src[baseSrc + 1];
          c = src[baseSrc + size];
          d = src[baseSrc + size + 1];
          dest[baseDest + x] = (a + b + c + d) * 0.25;
        }
      }
    }

    function lerpPixels(a, b, t) {
      var out = new Float64Array(a.length);
      var i;

      for (i = 0; i < a.length; i++) {
        out[i] = a[i] + (b[i] - a[i]) * t;
      }

      return out;
    }

    function createNoiseImage(seed) {
      var rng = createSeededRandom(seed || 1337);
      var image = new Float64Array(SIZE * SIZE * COLOR_CHANNELS);
      var i;

      for (i = 0; i < image.length; i++) {
        image[i] = rng() * 255;
      }

      return image;
    }

    function loadLennaImage() {
      return new Promise(function (resolve, reject) {
        var img = new Image();

        img.decoding = 'async';
        img.onload = function () {
          resolve(img);
        };
        img.onerror = reject;
        img.src = data.image;
      });
    }

    function loadPrecomputedMapping() {
      return global.fetch('walkthrough-mapping.json').then(function (response) {
        if (!response.ok) {
          throw new Error('Missing walkthrough-mapping.json.');
        }

        return response.json();
      });
    }

    function extractRgbPixelsFromImage(img) {
      var src;
      var pixels;
      var i;
      var p;

      offscreenCtx.clearRect(0, 0, SIZE, SIZE);
      offscreenCtx.drawImage(img, 0, 0, SIZE, SIZE);
      src = offscreenCtx.getImageData(0, 0, SIZE, SIZE).data;
      pixels = new Float64Array(SIZE * SIZE * COLOR_CHANNELS);

      for (i = 0, p = 0; i < src.length; i += 4, p += 3) {
        pixels[p] = src[i];
        pixels[p + 1] = src[i + 1];
        pixels[p + 2] = src[i + 2];
      }

      return pixels;
    }

    function renderPixels(pixels) {
      var dest = imageData.data;
      var i;
      var p;

      for (i = 0, p = 0; i < pixels.length; i += 3, p += 4) {
        dest[p] = clamp255(Math.round(pixels[i]));
        dest[p + 1] = clamp255(Math.round(pixels[i + 1]));
        dest[p + 2] = clamp255(Math.round(pixels[i + 2]));
        dest[p + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
    }

    function drawPixelBoxFromPixels(pixels, box) {
      var size = Math.max(1, Math.round(box.size));
      var patchImage;
      var dest;
      var x;
      var y;
      var srcBase;
      var patchBase;
      var c;

      patchCanvas.width = size;
      patchCanvas.height = size;
      patchImage = patchCtx.createImageData(size, size);
      dest = patchImage.data;

      for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
          srcBase = (((Math.round(box.y) + y) * SIZE) + (Math.round(box.x) + x)) * COLOR_CHANNELS;
          patchBase = (y * size + x) * 4;
          for (c = 0; c < COLOR_CHANNELS; c++) {
            dest[patchBase + c] = clamp255(Math.round(pixels[srcBase + c]));
          }
          dest[patchBase + 3] = 255;
        }
      }

      patchCtx.putImageData(patchImage, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(patchCanvas, box.x, box.y, box.size, box.size);
      ctx.restore();
    }

    function extractPatchPixels(pixels, box) {
      var size = Math.max(1, Math.round(box.size));
      var patch = new Float64Array(size * size * COLOR_CHANNELS);
      var x;
      var y;
      var srcBase;
      var destBase;
      var c;

      for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
          srcBase = (((Math.round(box.y) + y) * SIZE) + (Math.round(box.x) + x)) * COLOR_CHANNELS;
          destBase = (y * size + x) * COLOR_CHANNELS;
          for (c = 0; c < COLOR_CHANNELS; c++) {
            patch[destBase + c] = pixels[srcBase + c];
          }
        }
      }

      return patch;
    }

    function createCanvasFromPatch(pixels, size) {
      var bufferCanvas = document.createElement('canvas');
      var bufferCtx = bufferCanvas.getContext('2d');
      var patchImage;
      var dest;
      var i;
      var p;

      bufferCanvas.width = size;
      bufferCanvas.height = size;
      patchImage = bufferCtx.createImageData(size, size);
      dest = patchImage.data;

      for (i = 0, p = 0; i < pixels.length; i += COLOR_CHANNELS, p += 4) {
        dest[p] = clamp255(Math.round(pixels[i]));
        dest[p + 1] = clamp255(Math.round(pixels[i + 1]));
        dest[p + 2] = clamp255(Math.round(pixels[i + 2]));
        dest[p + 3] = 255;
      }

      bufferCtx.putImageData(patchImage, 0, 0);
      return bufferCanvas;
    }

    function writePatchPixels(targetPixels, box, patchPixels) {
      var size = Math.max(1, Math.round(box.size));
      var x;
      var y;
      var destBase;
      var srcBase;
      var c;

      for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
          destBase = (((Math.round(box.y) + y) * SIZE) + (Math.round(box.x) + x)) * COLOR_CHANNELS;
          srcBase = (y * size + x) * COLOR_CHANNELS;
          for (c = 0; c < COLOR_CHANNELS; c++) {
            targetPixels[destBase + c] = patchPixels[srcBase + c];
          }
        }
      }
    }

    function computeOrientedDomainPatch(pixels, node) {
      var sourceBox = getDomainBox(node.domainIndex);
      var sourceX = sourceBox.x;
      var sourceY = sourceBox.y;
      var channelDomain = new Float64Array(DOMAIN_AREA);
      var transformed = new Float64Array(DOMAIN_AREA);
      var patch = new Float64Array(DOMAIN_AREA * COLOR_CHANNELS);
      var x;
      var y;
      var c;
      var srcBase;
      var domainBase;
      var patchBase;

      for (c = 0; c < COLOR_CHANNELS; c++) {
        for (y = 0; y < DOMAIN_SIZE; y++) {
          for (x = 0; x < DOMAIN_SIZE; x++) {
            srcBase = (((sourceY + y) * SIZE) + (sourceX + x)) * COLOR_CHANNELS + c;
            domainBase = y * DOMAIN_SIZE + x;
            channelDomain[domainBase] = pixels[srcBase];
          }
        }

        transformBlock(channelDomain, 0, node.symmetry, DOMAIN_SIZE, transformed, 0);

        for (y = 0; y < DOMAIN_SIZE; y++) {
          for (x = 0; x < DOMAIN_SIZE; x++) {
            patchBase = (y * DOMAIN_SIZE + x) * COLOR_CHANNELS + c;
            patch[patchBase] = transformed[y * DOMAIN_SIZE + x];
          }
        }
      }

      return patch;
    }

    function getBranchNodes() {
      var nodes = [];

      if (data.levels[0] && data.levels[0][0]) {
        nodes.push(data.levels[0][0]);
      }

      if (data.levels[1] && data.levels[1][0]) {
        nodes.push(data.levels[1][0]);
      }

      if (data.levels[2] && data.levels[2][0]) {
        nodes.push(data.levels[2][0]);
      }

      return nodes;
    }

    function prepareIterationFrames() {
      var current;
      var iteration;

      iterationFrames = [randomPixels];
      current = randomPixels;

      for (iteration = 0; iteration < ITERATION_COUNT; iteration++) {
        current = global.FractalCodec.iterateColor(current, fullMapping);
        iterationFrames.push(current);
      }
    }

    function getFrame(stage) {
      if (stage.frame === 'lenna') {
        return lennaPixels;
      }
      if (stage.frame === 'random') {
        return randomPixels;
      }
      return iterationFrames[ITERATION_COUNT];
    }

    function rectMarkup(box, color, kind, options) {
      var fillOpacity = options && options.fillOpacity !== undefined ? options.fillOpacity : 0.16;
      var strokeOpacity = options && options.strokeOpacity !== undefined ? options.strokeOpacity : 0.95;
      var strokeWidth = options && options.strokeWidth !== undefined ? options.strokeWidth : (kind === 'range' ? 2.4 : 2.1);

      return [
        '<rect x="', box.x.toFixed(2), '" y="', box.y.toFixed(2),
        '" width="', box.size.toFixed(2), '" height="', box.size.toFixed(2),
        '" rx="0" ry="0" fill="', toRgba(color, fillOpacity),
        '" stroke="', toRgba(color, strokeOpacity),
        '" stroke-width="', strokeWidth.toFixed(2), '"></rect>'
      ].join('');
    }

    function arrowPathBetween(sourceBox, targetBox, key, options) {
      var sourceCenter = blockCenter(sourceBox);
      var targetCenter = blockCenter(targetBox);
      var dx = targetCenter.x - sourceCenter.x;
      var dy = targetCenter.y - sourceCenter.y;
      var dist = Math.sqrt(dx * dx + dy * dy) || 1;
      var ux = dx / dist;
      var uy = dy / dist;
      var sourceInset = (sourceBox.size * 0.5) / Math.max(Math.abs(ux), Math.abs(uy), 1e-6) + (options && options.sourcePad !== undefined ? options.sourcePad : 4);
      var targetInset = (targetBox.size * 0.5) / Math.max(Math.abs(ux), Math.abs(uy), 1e-6) + (options && options.targetPad !== undefined ? options.targetPad : 6);
      var source = {
        x: sourceCenter.x + ux * sourceInset,
        y: sourceCenter.y + uy * sourceInset
      };
      var target = {
        x: targetCenter.x - ux * targetInset,
        y: targetCenter.y - uy * targetInset
      };
      var sign;
      var nx;
      var ny;
      var bend;
      var shortLinkThreshold;
      var allowLoop;
      var loopBend;
      var loopSpan;
      var controlA;
      var controlB;
      var controlX;
      var controlY;

      dx = target.x - source.x;
      dy = target.y - source.y;
      dist = Math.sqrt(dx * dx + dy * dy) || 1;
      nx = -dy / dist;
      ny = dx / dist;
      sign = options && options.curveSign !== undefined ? options.curveSign : (hashString(key) % 2 === 0 ? 1 : -1);
      shortLinkThreshold = options && options.shortLinkThreshold !== undefined ? options.shortLinkThreshold : 34;
      allowLoop = !(options && options.allowLoop === false);

      if (allowLoop && dist < shortLinkThreshold) {
        loopBend = options && options.loopBend !== undefined ? options.loopBend : 42;
        loopSpan = options && options.loopSpan !== undefined ? options.loopSpan : 20;
        controlA = {
          x: target.x + nx * loopBend * sign + ux * loopSpan,
          y: target.y + ny * loopBend * sign + uy * loopSpan
        };
        controlB = {
          x: source.x + nx * loopBend * sign - ux * loopSpan,
          y: source.y + ny * loopBend * sign - uy * loopSpan
        };

        return [
          'M ', target.x.toFixed(2), ' ', target.y.toFixed(2),
          ' C ', controlA.x.toFixed(2), ' ', controlA.y.toFixed(2),
          ' ', controlB.x.toFixed(2), ' ', controlB.y.toFixed(2),
          ' ', source.x.toFixed(2), ' ', source.y.toFixed(2)
        ].join('');
      }

      bend = options && options.bend !== undefined ? options.bend : Math.max(14, Math.min(36, dist * 0.18 + (options && options.depth ? options.depth : 0) * 6));
      controlX = (source.x + target.x) * 0.5 + nx * bend * sign;
      controlY = (source.y + target.y) * 0.5 + ny * bend * sign;

      return [
        'M ', target.x.toFixed(2), ' ', target.y.toFixed(2),
        ' Q ', controlX.toFixed(2), ' ', controlY.toFixed(2),
        ' ', source.x.toFixed(2), ' ', source.y.toFixed(2)
      ].join('');
    }

    function overlayDefs() {
      return [
        '<defs>',
        '<marker id="walkthrough-arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="', DOMAIN_COLOR, '"></path>',
        '</marker>',
        '</defs>'
      ].join('');
    }

    function visibleNodesForStage(stage) {
      var maxDepth = stage.rootOnly ? 0 : stage.visibleDepth;

      return branchNodes.filter(function (node) {
        return node.depth <= maxDepth;
      });
    }

    function renderOverlay(stage) {
      var visibleNodes = visibleNodesForStage(stage);
      var rootNode = branchNodes[0];
      var parts = [overlayDefs()];
      var i;
      var node;
      var domainRect;
      var rangeRect;

      if (stage.rootOnly) {
        rangeRect = getRangeBox(rootNode.rangeIndex);
        parts.push(rectMarkup(rangeRect, RANGE_COLOR, 'range', {
          fillOpacity: 0.18,
          strokeWidth: 2.5
        }));

        if (stage.showDomain) {
          domainRect = getDomainBox(rootNode.domainIndex);
          parts.push(
            '<path d="', arrowPathBetween(domainRect, rangeRect, rootNode.path, { depth: 0 }),
            '" fill="none" stroke="', DOMAIN_COLOR,
            '" stroke-width="2.50" stroke-linecap="butt" opacity="0.96" marker-start="url(#walkthrough-arrow)"></path>'
          );
          parts.push(rectMarkup(domainRect, DOMAIN_COLOR, 'domain', {
            fillOpacity: 0.14,
            strokeWidth: 2.2
          }));
        }

        overlay.innerHTML = parts.join('');
        return;
      }

      for (i = 0; i < visibleNodes.length; i++) {
        node = visibleNodes[i];
        domainRect = getDomainBox(node.domainIndex);
        rangeRect = getRangeBox(node.rangeIndex);

        parts.push(
          '<path d="', arrowPathBetween(domainRect, rangeRect, node.path, { depth: node.depth }),
          '" fill="none" stroke="', DOMAIN_COLOR,
          '" stroke-width="2.50" stroke-linecap="butt" opacity="0.96" marker-start="url(#walkthrough-arrow)"></path>'
        );
      }

      for (i = 0; i < visibleNodes.length; i++) {
        node = visibleNodes[i];
        parts.push(rectMarkup(getDomainBox(node.domainIndex), DOMAIN_COLOR, 'domain', {
          fillOpacity: 0.14,
          strokeWidth: 2.2
        }));
      }

      for (i = 0; i < visibleNodes.length; i++) {
        node = visibleNodes[i];
        parts.push(rectMarkup(getRangeBox(node.rangeIndex), RANGE_COLOR, 'range', {
          fillOpacity: 0.18,
          strokeWidth: 2.5
        }));
      }

      overlay.innerHTML = parts.join('');
    }

    function drawMovingPatchTransition(sourceCanvas, orientedCanvas, targetCanvas, sourceBox, targetBox, symmetry, progress) {
      var eased = smoothstep(0, 1, progress);
      var sourceCenter = blockCenter(sourceBox);
      var targetCenter = blockCenter(targetBox);
      var centerX = sourceCenter.x + (targetCenter.x - sourceCenter.x) * eased;
      var centerY = sourceCenter.y + (targetCenter.y - sourceCenter.y) * eased;
      var size = sourceBox.size + (targetBox.size - sourceBox.size) * eased;
      var sourceOpacity = 1 - smoothstep(0.7, 0.96, eased);
      var orientedOpacity = smoothstep(0.16, 0.5, eased) * (1 - smoothstep(0.82, 0.97, eased));
      var targetOpacity = smoothstep(0.9, 0.995, eased);
      var domainOutlineOpacity = 0.8 * (1 - smoothstep(0.72, 0.97, eased));
      var rangeOutlineOpacity = 0.65 * orientedOpacity;
      var rotation = (symmetry & 3) * Math.PI * 0.5 * smoothstep(0.02, 0.9, eased);

      if (sourceOpacity > 0.001) {
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.globalAlpha = sourceOpacity;
        ctx.translate(centerX, centerY);
        ctx.rotate(rotation);
        ctx.drawImage(
          sourceCanvas,
          -size * 0.5,
          -size * 0.5,
          size,
          size
        );
        if (domainOutlineOpacity > 0.001) {
          ctx.strokeStyle = toRgba(DOMAIN_COLOR, domainOutlineOpacity);
          ctx.lineWidth = 1.6;
          ctx.strokeRect(
            -size * 0.5,
            -size * 0.5,
            size,
            size
          );
        }
        ctx.restore();
      }

      if (orientedOpacity > 0.001) {
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.globalAlpha = orientedOpacity;
        ctx.drawImage(
          orientedCanvas,
          centerX - size * 0.5,
          centerY - size * 0.5,
          size,
          size
        );
        if (rangeOutlineOpacity > 0.001) {
          ctx.strokeStyle = toRgba(RANGE_COLOR, rangeOutlineOpacity);
          ctx.lineWidth = 1.4;
          ctx.strokeRect(
            centerX - size * 0.5,
            centerY - size * 0.5,
            size,
            size
          );
        }
        ctx.restore();
      }

      if (targetOpacity > 0.001) {
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.globalAlpha = targetOpacity;
        ctx.drawImage(
          targetCanvas,
          centerX - size * 0.5,
          centerY - size * 0.5,
          size,
          size
        );
        ctx.restore();
      }
    }

    function renderAnimatedNodeFrame(basePixels, node, progress, sourceCanvas, orientedCanvas, targetCanvas) {
      var sourceBox = getDomainBox(node.domainIndex);
      var targetBox = getRangeBox(node.rangeIndex);

      renderPixels(basePixels);
      drawMovingPatchTransition(sourceCanvas, orientedCanvas, targetCanvas, sourceBox, targetBox, node.symmetry, progress);
      renderOverlay(STAGES[stageIndex]);
      updateChrome();
    }

    function renderRevealFrame(startPixels, endPixels, progress) {
      renderPixels(lerpPixels(startPixels, endPixels, smoothstep(0, 1, progress)));
      renderOverlay(STAGES[stageIndex]);
      updateChrome();
    }

    function runTimedAnimation(duration, onFrame, onDone) {
      var start = Date.now();

      function tick() {
        var elapsed;
        var progress;

        if (!isAnimating) {
          return;
        }

        elapsed = Date.now() - start;
        progress = Math.min(1, elapsed / duration);
        onFrame(progress);

        if (progress >= 1) {
          animationTimer = 0;
          onDone();
          return;
        }

        animationTimer = global.setTimeout(tick, ANIMATION_TICK_MS);
      }

      tick();
    }

    function updateChrome() {
      var stage = STAGES[stageIndex];

      caption.textContent = stage.caption;
      stepStat.textContent = 'Step ' + (stageIndex + 1) + ' / ' + STAGES.length + iterationDisplay;
      nextButton.disabled = !ready || isAnimating || stageIndex >= STAGES.length - 1;
      resetButton.disabled = !ready || isAnimating;
    }

    function renderCurrentStage() {
      var stage = STAGES[stageIndex];

      iterationDisplay = stage.frame === 'iter' ? ' · Iter ' + ITERATION_COUNT + '/' + ITERATION_COUNT : '';
      renderPixels(getFrame(stage));
      renderOverlay(stage);
      updateChrome();
    }

    function stopAnimation() {
      if (animationTimer) {
        global.clearTimeout(animationTimer);
        animationTimer = 0;
      }
      isAnimating = false;
    }

    function startIterationPlayback(nextIndex) {
      var iterationIndex = 1;

      stageIndex = nextIndex;
      isAnimating = true;
      iterationDisplay = ' · Iter 1/' + ITERATION_COUNT;
      renderPixels(iterationFrames[0]);
      renderOverlay(STAGES[stageIndex]);
      updateChrome();

      function animateIteration() {
        var currentPixels = iterationFrames[iterationIndex - 1];
        var nextPixels = iterationFrames[iterationIndex];
        var localPixels = new Float64Array(currentPixels);
        var nodeIndex = branchNodes.length - 1;

        iterationDisplay = ' · Iter ' + iterationIndex + '/' + ITERATION_COUNT;
        renderPixels(localPixels);
        renderOverlay(STAGES[stageIndex]);
        updateChrome();

        function animateNode() {
          var node;
          var targetBox;
          var targetPatch;
          var sourceCanvas;
          var orientedCanvas;
          var targetCanvas;

          if (nodeIndex < 0) {
            runTimedAnimation(FINAL_REVEAL_MS, function (progress) {
              renderRevealFrame(localPixels, nextPixels, progress);
            }, function () {
              if (iterationIndex >= ITERATION_COUNT) {
                stopAnimation();
                renderCurrentStage();
                return;
              }

              iterationIndex += 1;
              animationTimer = global.setTimeout(function () {
                animateIteration();
              }, ITERATION_GAP_MS);
            });
            return;
          }

          node = branchNodes[nodeIndex];
          targetBox = getRangeBox(node.rangeIndex);
          targetPatch = extractPatchPixels(nextPixels, targetBox);
          sourceCanvas = createCanvasFromPatch(
            extractPatchPixels(currentPixels, getDomainBox(node.domainIndex)),
            DOMAIN_SIZE
          );
          orientedCanvas = createCanvasFromPatch(
            computeOrientedDomainPatch(currentPixels, node),
            DOMAIN_SIZE
          );
          targetCanvas = createCanvasFromPatch(targetPatch, RANGE_SIZE);

          runTimedAnimation(NODE_ANIMATION_MS, function (progress) {
            renderAnimatedNodeFrame(localPixels, node, progress, sourceCanvas, orientedCanvas, targetCanvas);
          }, function () {
            writePatchPixels(localPixels, targetBox, targetPatch);
            renderPixels(localPixels);
            renderOverlay(STAGES[stageIndex]);
            updateChrome();
            nodeIndex -= 1;

            if (nodeIndex < 0) {
              animateNode();
              return;
            }

            animationTimer = global.setTimeout(function () {
              animateNode();
            }, NODE_GAP_MS);
          });
        }

        animateNode();
      }

      animateIteration();
    }

    function setUnavailable(message) {
      ready = false;
      caption.textContent = message;
      stepStat.textContent = 'Unavailable';
      nextButton.disabled = true;
      resetButton.disabled = true;
    }

    function initializeWalkthrough() {
      branchNodes = getBranchNodes();

      return Promise.all([loadLennaImage(), loadPrecomputedMapping()]).then(function (results) {
        var img = results[0];
        var mappingObject = results[1];

        if (!global.FractalCodec || !global.FractalCodec.iterateColor || !global.FractalCodec.pairsFromObject) {
          setUnavailable('Could not initialize walkthrough iterations.');
          return;
        }

        lennaPixels = extractRgbPixelsFromImage(img);
        fullMapping = global.FractalCodec.pairsFromObject(mappingObject);

        randomPixels = createNoiseImage(1337);
        prepareIterationFrames();
        ready = true;
        stageIndex = 0;
        renderCurrentStage();
      }).catch(function (error) {
        console.error(error);
        setUnavailable(error && error.message ? error.message : 'Could not load Lenna.');
      });
    }

    nextButton.addEventListener('click', function () {
      var nextIndex;

      if (!ready || isAnimating || stageIndex >= STAGES.length - 1) {
        return;
      }

      nextIndex = stageIndex + 1;

      if (STAGES[nextIndex].frame === 'iter') {
        startIterationPlayback(nextIndex);
        return;
      }

      stageIndex = nextIndex;
      renderCurrentStage();
    });

    resetButton.addEventListener('click', function () {
      if (!ready || isAnimating) {
        return;
      }

      stopAnimation();
      iterationDisplay = '';
      stageIndex = 0;
      renderCurrentStage();
    });

    initializeWalkthrough();
  });
})(window);
