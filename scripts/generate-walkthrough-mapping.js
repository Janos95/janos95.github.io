#!/usr/bin/env node

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const repoRoot = path.resolve(__dirname, '..');
const codec = require(path.join(repoRoot, 'blog/fractal-compression/fractal.js'));

const inputPath = path.resolve(
  process.argv[2] || path.join(repoRoot, 'blog/fractal-compression/lenna.png')
);
const outputPath = path.resolve(
  process.argv[3] || path.join(repoRoot, 'blog/fractal-compression/walkthrough-mapping.json')
);

function fail(message) {
  console.error(message);
  process.exit(1);
}

function encodeImageToRawRgb(imagePath) {
  const result = spawnSync(
    'magick',
    [imagePath, '-resize', '512x512!', '-depth', '8', 'rgb:-'],
    { encoding: null, maxBuffer: 4 * 1024 * 1024 }
  );

  if (result.error) {
    fail('Failed to run ImageMagick `magick`: ' + result.error.message);
  }

  if (result.status !== 0) {
    fail((result.stderr || Buffer.from('')).toString() || 'Image conversion failed.');
  }

  return result.stdout;
}

function main() {
  let rawRgb;
  let pixels;
  let mapping;
  let mappingObject;
  let i;

  if (!fs.existsSync(inputPath)) {
    fail('Input image not found: ' + inputPath);
  }

  rawRgb = encodeImageToRawRgb(inputPath);

  if (rawRgb.length !== codec.IMAGE_SIZE * codec.IMAGE_SIZE * codec.COLOR_CHANNELS) {
    fail('Unexpected converted image size: ' + rawRgb.length + ' bytes.');
  }

  pixels = new Float64Array(rawRgb.length);
  for (i = 0; i < rawRgb.length; i++) {
    pixels[i] = rawRgb[i];
  }

  mapping = codec.compressColorFromPixels(pixels);
  mappingObject = codec.pairsToObject(mapping);

  fs.writeFileSync(outputPath, JSON.stringify(mappingObject));
  console.log('Wrote ' + outputPath);
}

main();
