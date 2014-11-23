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

    // Size of vocabulary
    2: i32 vocab_size,

    // Vocabulary index of start-of-sentence token
    3: i32 start_token_index,

    // Vocabulary index of end-of-sentence token
    4: i32 end_token_index,

    // Word vector dimension
    5: i32 wordvec_dim,

    // Hidden layer dimension
    6: i32 hidden_dim,
}

/**
 * Model parameters as flat vectors of doubles.
 */
struct Params {
    1: list<double> wordvec_weights,
    2: list<double> input_hidden_weights,
    3: list<double> input_hidden_biases,
    4: list<double> hidden_output_weights,
    5: list<double> hidden_output_biases,
}

/**
 * Server information.
 */
struct ServerInfo {
    1: string ip,
    2: i32 port,
}

struct AnnounceRequest {
    1: ServerInfo worker_info,
}

struct AnnounceResponse {
    1: ModelInfo model_info,
    2: Params params,
    3: list<string> shard_paths,
    4: double learn_rate,
    5: list<ServerInfo> param_servers,
}

struct UpdateRequest {
    1: Params update,
    2: optional ServerInfo worker_info,
}

struct UpdateResponse {
    1: optional list<ServerInfo> param_servers,
}

struct PullRequest {
    1: optional ServerInfo worker_info,
}

struct PullResponse {
    1: Params params,
    2: optional list<ServerInfo> param_servers,
}

service ParamService {
    // Announce a worker to the master
    AnnounceResponse announce(1:AnnounceRequest request),

    // Push a parameter update to the master
    UpdateResponse push_update(1:UpdateRequest request),

    // Request up-to-date parameters from the master
    PullResponse pull_params(1:PullRequest request),
}


/* -----------------------------
 *  Worker service definition
 * ----------------------------- */

struct HBRequest {
    1: optional ServerInfo master_info,
}

struct HBResponse {
    1: optional ServerInfo worker_info,
}

/**
 * Request to the worker to start computation.
 */
struct StartRequest {
    // Paths to dataset shard files on disk
    1: list<string> shard_paths,

    // Learning rate
    2: double learn_rate
}

struct ReassignRequest {
    1: list<string> shard_paths,
}

/**
 * A worker node that computes on one data shard.
 */
service WorkerService {
    // A heartbeat to check if worker is alive
    HBResponse heartbeat(1:HBRequest request),

    // Start computation on a set of shards
    void start(1:StartRequest request),

    // Stop computation
    void stop(),

    // Reassign shards to worker
    void reassign(1:ReassignRequest request),
}
