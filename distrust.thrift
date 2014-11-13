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

struct StartRequest {
    1: string dataset_path,
    2: string label_path,
    3: i32 input_dim,
    4: i32 start_line,
    5: i32 shard_lines,
    6: double learn_rate,
}

/**
 * A worker node that computes on one data shard.
 */
service Worker {
    HBResponse heartbeat(1:HBRequest request),
    void start(1:StartRequest request),
    void stop(),
}
