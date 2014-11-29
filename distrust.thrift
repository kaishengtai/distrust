namespace cpp distrust

/* --------------------------------------
 *  Parameter server service definition
 * -------------------------------------- */

/**
 * Information on the structure of the model.
 */
struct ModelInfo {
    // Language model context window size
    1: i32 window_size,

    // Word vector dimension
    2: i32 wordvec_dim,

    // Hidden layer dimension
    3: i32 hidden_dim,

    // Vocabulary index of start-of-sentence token
    4: i32 start_token_index,

    // Vocabulary index of end-of-sentence token
    5: i32 end_token_index,

    // Vocabulary index of unknown-word token
    6: i32 unk_token_index,

    // Vocabulary
    7: list<string> vocab,
}

/**
 * Model parameters as flat vectors of doubles.
 */
struct Params {
    1: list<list<double>> wordvec_w,
    2: list<list<double>> input_hidden_w,
    3: list<double> input_hidden_b,
    4: list<double> hidden_output_w,
    5: list<double> hidden_output_b,
}

struct ParamUpdate {
    1: map<i32, list<double>> wordvec_w,
    2: list<list<double>> input_hidden_w,
    3: list<double> input_hidden_b,
    4: list<double> hidden_output_w,
    5: list<double> hidden_output_b,
}

struct AnnounceResponse {
    1: ModelInfo model_info,
    2: Params params,
    3: list<string> shard_paths,
    4: double learn_rate,
    5: i32 batch_size,
}

service ParamService {
    // Announce a worker to the master
    AnnounceResponse announce(1:i32 worker_port),

    // Push a parameter update to the master
    void push_update(1:ParamUpdate update),

    // Request up-to-date parameters from the master
    Params pull_params(),
}


/* -----------------------------
 *  Worker service definition
 * ----------------------------- */

struct HBResponse {
    1: list<string> completed_shards,
}

/**
 * Request to the worker to start computation.
 */
struct StartRequest {
    // Paths to dataset shard files on disk
    1: list<string> shard_paths,

    // Initial learning rate
    2: double learn_rate,

    // Minibatch size for training
    3: i32 batch_size,
}

/**
 * A worker node that computes on one data shard.
 */
service WorkerService {
    // A heartbeat to check if worker is alive
    HBResponse heartbeat(),

    // Start computation on a set of shards
    void start(1:StartRequest request),

    // Stop computation
    void stop(),

    // Reassign shards to worker
    void reassign(1:list<string> shard_paths),
}
