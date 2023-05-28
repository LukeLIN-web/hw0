#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
 // Iterate through batches
    for (size_t i = 0; i < m; i += batch)
    {
        // Get the current batch
        size_t batch_size = std::min(batch, m - i);
        const float *X_batch = X + i * n;
        const unsigned char *y_batch = y + i;

        // Compute the logits for the batch
        float *logits = new float[batch_size * k];
        for (size_t j = 0; j < batch_size; j++)
        {
            for (size_t c = 0; c < k; c++)
            {
                float dot_product = 0.0;
                for (size_t d = 0; d < n; d++)
                {
                    dot_product += X_batch[j * n + d] * theta[d * k + c];
                }
                logits[j * k + c] = dot_product;
            }
        }

        // Compute the softmax probabilities
        float *softmax_probs = new float[batch_size * k];
        for (size_t j = 0; j < batch_size; j++)
        {
            float max_logit = logits[j * k];
            for (size_t c = 1; c < k; c++)
            {
                max_logit = std::max(max_logit, logits[j * k + c]);
            }

            float sum_exp_logit = 0.0;
            for (size_t c = 0; c < k; c++)
            {
                softmax_probs[j * k + c] = std::exp(logits[j * k + c] - max_logit);
                sum_exp_logit += softmax_probs[j * k + c];
            }

            for (size_t c = 0; c < k; c++)
            {
                softmax_probs[j * k + c] /= sum_exp_logit;
            }
        }

        // Compute the gradient of the loss with respect to logits
        float *grad_logits = new float[batch_size * k];
        for (size_t j = 0; j < batch_size; j++)
        {
            for (size_t c = 0; c < k; c++)
            {
                grad_logits[j * k + c] = softmax_probs[j * k + c];
                if (c == y_batch[j])
                {
                    grad_logits[j * k + c] -= 1.0;
                }
            }
        }

        // Compute the gradient of the loss with respect to theta
        float *grad_theta = new float[n * k];
        for (size_t d = 0; d < n; d++)
        {
            for (size_t c = 0; c < k; c++)
            {
                float dot_product = 0.0;
                for (size_t j = 0; j < batch_size; j++)
                {
                    dot_product += X_batch[j * n + d] * grad_logits[j * k + c];
                }
                grad_theta[d * k + c] = dot_product / batch_size;
            }
        }
              // Update theta using gradient descent
        for (size_t d = 0; d < n; d++)
        {
            for (size_t c = 0; c < k; c++)
            {
            theta[d * k + c] -= lr * grad_theta[d * k + c];
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
