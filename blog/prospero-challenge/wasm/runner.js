(function () {
    var button = document.querySelector('[data-prospero-run]');
    var resetButton = document.querySelector('[data-prospero-reset]');
    var status = document.querySelector('[data-prospero-status]');
    var output = document.querySelector('[data-prospero-output]');
    var canvas = document.querySelector('[data-prospero-canvas]');
    var modulePromise = null;
    var loaded = false;
    var activeMode = 'single-thread';
    var fallbackReason = '';
    var activeThreads = 1;
    var reportedThreads = navigator.hardwareConcurrency || 0;

    function setStatus(text) {
        if (status) {
            status.textContent = text;
        }
    }

    function decodeBase64(value) {
        var binary = atob(value);
        var bytes = new Uint8Array(binary.length);
        for (var i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    function hasThreadSupport() {
        if (typeof SharedArrayBuffer === 'undefined') {
            fallbackReason = 'SharedArrayBuffer unavailable';
            return false;
        }
        if (typeof createProsperoOmpModule !== 'function') {
            fallbackReason = 'threaded module unavailable';
            return false;
        }
        fallbackReason = '';
        return true;
    }

    function getModule() {
        if (!modulePromise) {
            var useThreads = hasThreadSupport();
            activeThreads = useThreads ? 8 : 1;
            activeMode = useThreads ? activeThreads + '-thread OpenMP' : 'single-thread';
            var factory = useThreads ? createProsperoOmpModule : createProsperoModule;
            var binary = useThreads ? window.PROSPERO_OMP_WASM_BASE64 : window.PROSPERO_WASM_BASE64;

            modulePromise = factory({
                wasmBinary: decodeBase64(binary),
                print: function () {},
                printErr: function (message) {
                    console.error(message);
                }
            });
        }
        return modulePromise;
    }

    function formatMs(value) {
        return value.toFixed(3) + ' ms';
    }

    function clearCanvas() {
        if (!canvas) {
            return;
        }
        var context = canvas.getContext('2d');
        context.fillStyle = '#000000';
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    function drawImage(module) {
        if (!canvas) {
            return;
        }
        var context = canvas.getContext('2d');
        var pointer = module.ccall('prospero_image_data', 'number', [], []);
        var pixels = module.HEAPU8.subarray(pointer, pointer + canvas.width * canvas.height);
        var imageData = context.createImageData(canvas.width, canvas.height);

        for (var y = 0; y < canvas.height; y++) {
            var sourceRow = canvas.height - 1 - y;
            for (var x = 0; x < canvas.width; x++) {
                var value = pixels[sourceRow * canvas.width + x];
                var target = (y * canvas.width + x) * 4;
                imageData.data[target] = value;
                imageData.data[target + 1] = value;
                imageData.data[target + 2] = value;
                imageData.data[target + 3] = 255;
            }
        }

        context.putImageData(imageData, 0, 0);
    }

    function callWithString(module, name, value) {
        var bytes = module.lengthBytesUTF8(value) + 1;
        var ptr = module._malloc(bytes);
        if (!ptr) {
            throw new Error('Unable to allocate WASM memory');
        }
        try {
            module.stringToUTF8(value, ptr, bytes);
            return module['_' + name](ptr);
        } finally {
            module._free(ptr);
        }
    }

    async function runBenchmark() {
        if (!button || !output) {
            return;
        }

        button.disabled = true;
        setStatus(loaded ? 'Running benchmark...' : 'Loading WASM...');

        try {
            var module = await getModule();

            if (!loaded) {
                var instructionCount = callWithString(module, 'prospero_load', window.PROSPERO_VM_TEXT);
                if (instructionCount <= 0) {
                    throw new Error('Failed to load prospero.vm');
                }
                loaded = true;
            }

            if (activeThreads > 1) {
                module.ccall('prospero_set_threads', null, ['number'], [activeThreads]);
            }

            var mean = module.ccall('prospero_benchmark', 'number', ['number', 'number'], [4, 24]);
            var std = module.ccall('prospero_std_ms', 'number', [], []);
            var min = module.ccall('prospero_min_ms', 'number', [], []);
            var max = module.ccall('prospero_max_ms', 'number', [], []);
            var instructions = module.ccall('prospero_instructions', 'number', [], []);
            var frontier = module.ccall('prospero_frontier_terms', 'number', [], []);
            var checksum = module.ccall('prospero_checksum', 'number', [], []);
            drawImage(module);

            output.innerHTML = [
                '<div><strong>Mode:</strong> ' + activeMode + (fallbackReason ? ' (' + fallbackReason + ')' : '') + '</div>',
                '<div><strong>Mean:</strong> ' + formatMs(mean) + '</div>',
                '<div><strong>Std dev:</strong> ' + formatMs(std) + '</div>',
                '<div><strong>Min:</strong> ' + formatMs(min) + '</div>',
                '<div><strong>Max:</strong> ' + formatMs(max) + '</div>',
                '<div><strong>Instructions:</strong> ' + instructions.toLocaleString() + '</div>',
                '<div><strong>Frontier terms:</strong> ' + frontier.toLocaleString() + '</div>',
                '<div><strong>Browser cores:</strong> ' + (reportedThreads || 'unknown') + '</div>',
                '<div><strong>Checksum:</strong> ' + checksum.toLocaleString() + '</div>'
            ].join('');
            setStatus('Benchmark complete.');
            button.textContent = 'Re-evaluate';
        } catch (error) {
            console.error(error);
            setStatus('Benchmark failed.');
            output.textContent = error && error.message ? error.message : 'Unknown error';
        } finally {
            button.disabled = false;
        }
    }

    function resetBenchmark() {
        clearCanvas();
        if (output) {
            output.textContent = '';
        }
        if (button) {
            button.textContent = 'Run benchmark';
        }
        setStatus('Runs the 1024x1024 Prospero renderer in WebAssembly.');
    }

    if (button) {
        button.addEventListener('click', runBenchmark);
    }
    if (resetButton) {
        resetButton.addEventListener('click', resetBenchmark);
    }
    clearCanvas();
    if (!hasThreadSupport()) {
        setStatus('Runs the 1024x1024 renderer in single-thread WebAssembly. Fast OpenMP mode needs SharedArrayBuffer.');
    }
}());
