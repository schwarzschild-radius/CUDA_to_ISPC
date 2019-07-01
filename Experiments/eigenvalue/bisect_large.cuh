#ifndef _BISECT_LARGE_CUH_
#define _BISECT_LARGE_CUH_
    ////////////////////////////////////////////////////////////////////////////////
    //! Run the kernels to compute the eigenvalues for large matrices
    //! @param  input   handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size  matrix size
    //! @param  precision  desired precision of eigenvalues
    //! @param  lg  lower limit of Gerschgorin interval
    //! @param  ug  upper limit of Gerschgorin interval
    //! @param  iterations  number of iterations (for timing)
    ////////////////////////////////////////////////////////////////////////////////
    void
    computeEigenvaluesLargeMatrix(const InputData &input, const ResultDataLarge &result,
                                  const unsigned int mat_size, const float precision,
                                  const float lg, const float ug,
                                  const unsigned int iterations);

    ////////////////////////////////////////////////////////////////////////////////
    //! Initialize variables and memory for result
    //! @param  result handles to memory
    //! @param  matr_size  size of the matrix
    ////////////////////////////////////////////////////////////////////////////////
    void
    initResultDataLargeMatrix(ResultDataLarge &result, const unsigned int mat_size);

    ////////////////////////////////////////////////////////////////////////////////
    //! Cleanup result memory
    //! @param result  handles to memory
    ////////////////////////////////////////////////////////////////////////////////
    void
    cleanupResultDataLargeMatrix(ResultDataLarge &result);

    ////////////////////////////////////////////////////////////////////////////////
    //! Process the result, that is obtain result from device and do simple sanity
    //! checking
    //! @param  input   handles to input data
    //! @param  result  handles to result data
    //! @param  mat_size  matrix size
    //! @param  filename  output filename
    ////////////////////////////////////////////////////////////////////////////////
    bool
    processResultDataLargeMatrix(const InputData &input, const ResultDataLarge &result,
                                 const unsigned int mat_size,
                                 const char *filename,
                                 const unsigned int user_defined, char *exec_path);
#endif