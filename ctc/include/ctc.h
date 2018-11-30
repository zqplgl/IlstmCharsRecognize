#pragma once

typedef struct CUstream_st* CUstream;

typedef enum {
    CTC_STATUS_SUCCESS = 0,
    CTC_STATUS_MEMOPS_FAILED = 1,
    CTC_STATUS_INVALID_VALUE = 2,
    CTC_STATUS_EXECUTION_FAILED = 3,
    CTC_STATUS_UNKNOWN_ERROR = 4
} ctcStatus_t;



typedef enum {
    CTC_CPU = 0,
    CTC_GPU = 1
} ctcComputeLocation;

struct ctcOptions {
    ctcComputeLocation loc;
    union {
        unsigned int num_threads;
        CUstream stream;
    };

    int blank_label;
};
