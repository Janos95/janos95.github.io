#include "vm.h"

#include <emscripten/emscripten.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
  char key[32];
  int idx;
} IdMap;

static IdMap ids[MAX_INS];
static int nids;
static double last_mean, last_std, last_min, last_max;
static Worker bench_worker;
static int thread_count = 8;

static int parse_op(const char *s) {
  static const char *name[] = {"const", "var-x", "var-y", "add", "sub", "mul", "neg", "square", "sqrt", "min", "max"};
  for (int i = 0; i < (int)(sizeof name / sizeof name[0]); i++)
    if (!strcmp(s, name[i])) return i;
  return -1;
}

static int idof(const char *s) {
  for (int i = 0; i < nids; i++)
    if (!strcmp(ids[i].key, s)) return ids[i].idx;
  return -1;
}

static void parse_arg(const char *s, int16_t *ref, float *value) {
  if (s[0] == '_') *ref = idof(s);
  else *value = strtof(s, NULL);
}

EMSCRIPTEN_KEEPALIVE
void prospero_set_threads(int threads) {
  if (threads < 1) threads = 1;
  if (threads > 8) threads = 8;
  thread_count = threads;
}

EMSCRIPTEN_KEEPALIVE
int prospero_load(const char *text) {
  vm_init_coords();
  vm_reset_program();
  nids = 0;

  const char *p = text;
  char line[256], id[32], op[32], a[64], b[64];
  while (*p) {
    int len = 0;
    while (p[len] && p[len] != '\n' && len < (int)sizeof(line) - 1) len++;
    memcpy(line, p, len);
    line[len] = '\0';
    while (p[len] && p[len] != '\n') len++;
    p += len + (p[len] == '\n');

    a[0] = b[0] = 0;
    int n = sscanf(line, "%31s %31s %63s %63s", id, op, a, b);
    if (n < 2 || line[0] == '#') continue;

    int parsed_op = parse_op(op);
    if (parsed_op < 0) return -1;

    int idx = vm_instruction_count();
    if (idx >= MAX_INS || nids >= MAX_INS) return -2;
    strcpy(ids[nids].key, id);
    ids[nids++].idx = idx;

    Instr in = {(Op)parsed_op, -1, -1, 0, 0};
    if (n >= 3) parse_arg(a, &in.a, &in.fa);
    if (n >= 4) parse_arg(b, &in.b, &in.fb);
    if (vm_add_instr(in) != idx) return -3;
  }

  return vm_instruction_count();
}

EMSCRIPTEN_KEEPALIVE
double prospero_benchmark(int warmups, int runs) {
#ifdef _OPENMP
  omp_set_num_threads(thread_count);

  double t0 = 0;
  last_mean = 0;
  last_std = 0;
  last_min = 1e9;
  last_max = 0;

#pragma omp parallel
  {
    Worker w;
    vm_worker_init(&w);
    for (int r = 0; r < warmups; r++) vm_render_frame(&w);
    for (int r = 0; r < runs; r++) {
#pragma omp single
      t0 = emscripten_get_now();
      vm_render_frame(&w);
#pragma omp single
      {
        double dt = emscripten_get_now() - t0;
        last_mean += dt;
        last_std += dt * dt;
        if (dt < last_min) last_min = dt;
        if (dt > last_max) last_max = dt;
      }
    }
  }

  last_mean /= runs;
  last_std = sqrt(fmax(0, last_std / runs - last_mean * last_mean));
  return last_mean;
#else
  vm_worker_init(&bench_worker);

  for (int r = 0; r < warmups; r++) vm_render_frame(&bench_worker);

  double sum = 0;
  double sum_sq = 0;
  last_min = 1e9;
  last_max = 0;

  for (int r = 0; r < runs; r++) {
    double t0 = emscripten_get_now();
    vm_render_frame(&bench_worker);
    double dt = emscripten_get_now() - t0;
    sum += dt;
    sum_sq += dt * dt;
    if (dt < last_min) last_min = dt;
    if (dt > last_max) last_max = dt;
  }

  last_mean = sum / runs;
  last_std = sqrt(fmax(0, sum_sq / runs - last_mean * last_mean));
  return last_mean;
#endif
}

EMSCRIPTEN_KEEPALIVE int prospero_instructions(void) { return vm_instruction_count(); }
EMSCRIPTEN_KEEPALIVE int prospero_frontier_terms(void) { return vm_frontier_count(); }
EMSCRIPTEN_KEEPALIVE int prospero_checksum(void) { return vm_checksum(); }
EMSCRIPTEN_KEEPALIVE const uint8_t *prospero_image_data(void) { return vm_image_data(); }
EMSCRIPTEN_KEEPALIVE double prospero_std_ms(void) { return last_std; }
EMSCRIPTEN_KEEPALIVE double prospero_min_ms(void) { return last_min; }
EMSCRIPTEN_KEEPALIVE double prospero_max_ms(void) { return last_max; }
