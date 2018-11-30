#pragma once

#include "ctc.h"

namespace CTC {

int get_warpctc_version();

const char* ctcGetStatusString(ctcStatus_t status);

template<typename Dtype>
ctcStatus_t compute_ctc_loss_cpu(const Dtype* const activations,
                             Dtype* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             Dtype *costs,
                             void *workspace,
                             ctcOptions options);

template<typename Dtype>
ctcStatus_t compute_ctc_loss_gpu(const Dtype* const activations,
	Dtype* gradients,
	const int* const flat_labels,
	const int* const label_lengths,
	const int* const input_lengths,
	int alphabet_size,
	int minibatch,
	Dtype *costs,
	void *workspace,
	ctcOptions options);


template<typename Dtype>
ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions info,
                               size_t* size_bytes);



} // namespace ctc
