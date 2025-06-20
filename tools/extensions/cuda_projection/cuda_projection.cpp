
#include <torch/extension.h>
#include <iostream>
#include <math.h> 
// cuda render utils

using namespace std;
using namespace torch::indexing;




#include <torch/extension.h>
#include <vector>

// Function to project points
torch::Tensor projection_fw(torch::Tensor verts, torch::Tensor Ps, int H = -1, int W = -1) {
    // Check the input dimensions
    TORCH_CHECK(verts.dim() == 3, "verts should be a 3D tensor");
    TORCH_CHECK(Ps.dim() == 4, "Ps should be a 4D tensor");

    int B = verts.size(0);
    int N = verts.size(1);

    // Add a column of ones to verts
    auto ones = torch::ones({B, N, 1}, verts.options());
    verts = torch::cat({verts, ones}, -1); // B, N, 4

    // Perform the matrix multiplication
    verts = torch::einsum("bvm,bcmn ->bcvn", {verts, Ps.transpose(2, 3)});


    // Normalize by the z component
    verts.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0},
                     verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}) / verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2}));
    verts.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1},
                     verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1}) / verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2}));

    // If H and W are greater than 0, convert to normalized device coordinates (NDC)
    if (H > 0 && W > 0) {
        verts.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0},
                         2 * verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}) / W - 1);
        verts.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1},
                         2 * verts.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1}) / H - 1);
    }

    return verts;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("projection_fw", &projection_fw, "projection_fw",
          py::arg("verts"), py::arg("Ps"), py::arg("H") = -1, py::arg("W") = -1);
}