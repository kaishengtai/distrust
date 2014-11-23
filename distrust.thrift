namespace cpp distrust

/**
 * Status of the parameter servers.
 */
struct Status {
    1: i32 leader_id,
    2: list<i32> param_server_ids,
}

struct HBRequest {
    1: optional Status status,
}

struct HBResponse {
    1: optional i32 worker_id,
}

/**
 * Request to the worker to start computation.
 */
struct StartRequest {
    // Path to the dataset shard file on disk
    1: string shard_path,

    // Initial learning rate
    2: double learn_rate
}

/**
 * A worker node that computes on one data shard.
 */
service WorkerService {
    HBResponse heartbeat(1:HBRequest request),
    void start(1:StartRequest request),
    void stop(),
}
