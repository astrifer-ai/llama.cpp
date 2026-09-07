// Dump the QSA indexer top-k selection during DECODE, to measure whether the selected cells
// cluster. That single question decides whether flash attention can be fixed cheaply (skip
// fully-masked tiles in fattn-vec.cuh, numerics untouched, bit-identical) or expensively
// (physically gather the selected K/V, which reorders the softmax rescale and forfeits
// bit-identity). Flash attention measures 61.6 ms/step -- 46.6 % of GPU busy at 131K x 4 --
// so the difference between the two is days of work and most of the prize.
//
// Decode specifically: during prefill n_tps is the ubatch size and the tensor is one row per
// query token, but the 61.6 ms is spent in decode where n_tps = 1. The callback is therefore
// held OFF during prefill and switched on for the generated tokens only.
//
// The dumped values are indices into the [0, n_kv) cell range -- the same index space the
// kq_mask uses and the same one fattn-vec.cuh's k_VKQ_0 loop strides over -- so an aligned
// tile of `nthreads` (128 on this hardware) cells is fully masked iff no dumped index for
// that (layer, stream, step) falls inside it.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dump_state {
    FILE *      f      = nullptr;
    std::string prefix = "indexer_top_k";
    bool        active = false;   // off during prefill, on during generation
    int32_t     step   = -1;
    long        n      = 0;
};

static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    dump_state * st = (dump_state *) ud;
    if (!st->active) {
        return false;
    }
    const bool match = strncmp(t->name, st->prefix.c_str(), st->prefix.size()) == 0;
    if (ask) {
        return match;
    }
    if (!match) {
        return true;
    }
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    char name[64] = {0};
    strncpy(name, t->name, 63);
    int32_t type = (int32_t) t->type;
    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    fwrite(name,      1, 64, st->f);
    fwrite(&type,     4, 1,  st->f);
    fwrite(ne,        8, 4,  st->f);
    fwrite(&st->step, 4, 1,  st->f);
    fwrite(buf.data(), 1, nb, st->f);
    st->n++;
    return true;
}

static llama_token greedy(llama_context * ctx, const llama_vocab * vocab) {
    const int n = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, -1);
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (lg[i] > lg[best]) best = i;
    }
    return (llama_token) best;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    dump_state st;
    const char * out = getenv("QSA_DUMP_OUT");
    if (!out) { fprintf(stderr, "set QSA_DUMP_OUT=<file>\n"); return 1; }
    st.f = fopen(out, "wb");
    if (!st.f) { fprintf(stderr, "cannot open %s\n", out); return 1; }
    if (const char * p = getenv("QSA_DUMP_PREFIX")) st.prefix = p;
    const long ntok_want = getenv("QSA_DUMP_NTOK") ? atol(getenv("QSA_DUMP_NTOK")) : 0;

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval           = dump_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (toks.empty()) {
        fprintf(stderr, "empty prompt -- pass -f with a corpus file\n");
        return 1;
    }
    if (ntok_want > 0) {
        // NB: tile the prompt from a COPY. Inserting from toks into toks invalidates the
        // source iterators the moment the vector reallocates, which is undefined behaviour
        // and would corrupt the prompt at exactly the long lengths this tool exists for.
        const std::vector<llama_token> base = toks;
        while ((long) toks.size() < ntok_want) {
            toks.insert(toks.end(), base.begin(), base.end());
        }
        toks.resize(ntok_want);
    }
    LOG_INF("qsa-topk-dump: %zu prompt tokens, generating %d, dumping '%s' to %s\n",
            toks.size(), params.n_predict, st.prefix.c_str(), out);

    // prefill with the callback OFF
    st.active = false;
    const int nb = params.n_batch > 0 ? params.n_batch : 512;
    for (size_t i = 0; i < toks.size(); i += nb) {
        const int n = (int) std::min<size_t>(nb, toks.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n))) {
            fprintf(stderr, "prefill failed at %zu\n", i); return 1;
        }
    }

    // generate with the callback ON -- these are the steps that cost 61.6 ms each
    st.active = true;
    llama_token cur = greedy(ctx, vocab);
    for (int s = 0; s < params.n_predict; ++s) {
        st.step = s;
        if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) break;
        cur = greedy(ctx, vocab);
    }
    fclose(st.f);
    LOG_INF("qsa-topk-dump: wrote %ld tensor dumps\n", st.n);
    llama_backend_free();
    return 0;
}
