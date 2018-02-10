using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    [TestClass]
    public partial class MatrixOperationsTest
    {
        double[][] _matrix1, _matrix2;
        MatrixOperations _mo;

        public MatrixOperationsTest()
        {
            _mo = new MatrixOperations();
        }

        #region Data assignment

      
        protected void InitData_dataset_2_rows()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 1, 2 };
            _matrix1[1] = new double[] { 3, 4 };

            _matrix2 = new double[2][];
            _matrix2[0] = new double[] { 5, 6 };
            _matrix2[1] = new double[] { 7, 8 };
        }

        /// <summary>
        /// Example from:
        /// https://math.la.asu.edu/~gardner/QR.pdf
        /// </summary>
        protected void InitData_dataset_2_rows_householder()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 3, 4 };
            _matrix1[1] = new double[] { 4, 0 };

           
        }

        /// <summary>
        /// Example taken from:
        /// https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
        /// R:
        /// A = matrix(c(2, 1, 1, 2),nrow=2,ncol=2)
        /// </summary>
        protected void InitData_dataset_2_rows_symmetric_eigenValue()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 2, 1 };
            _matrix1[1] = new double[] { 1, 2 };

            _matrix2 = new double[2][];
            _matrix2[0] = new double[] { 5, 6 };
            _matrix2[1] = new double[] { 7, 8 };
        }

        /// <summary>
        /// /// R:
        /// A = matrix(c(1, 1, 1, 1),nrow=2,ncol=2)
        /// </summary>
        protected void InitData_dataset_2_rows_allEqual_eigenValue()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 1, 1 };
            _matrix1[1] = new double[] { 1, 1 };            
        }

        /// <summary>
        /// R:
        /// A = matrix(c(1,2,3,4,5,6,7,8,9),nrow=3,ncol=3)
        /// </summary>
        protected void InitData_dataset_3_rows_non_symmetric()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1, 2, 3 };
            _matrix1[1] = new double[] { 4, 5, 6 };
            _matrix1[2] = new double[] { 7, 8, 9 };
        }



        /// <summary>
        /// Dataset taken from:
        ///
        /// https://en.wikipedia.org/wiki/Givens_rotation
        /// </summary>
        protected void InitData_dataset_3_rows_symmetric_givens_rotation()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 6, 5, 0 };
            _matrix1[1] = new double[] { 5, 1, 4 };
            _matrix1[2] = new double[] { 0, 4, 3 };
        }

        /// <summary>
        /// Example from:
        /// https://math.la.asu.edu/~gardner/QR.pdf
        /// R:
        /// A = matrix(c(1,0,1,0,1,2,1,2,0),nrow=3,ncol=3)
        /// </summary>
        protected void InitData_dataset_3_rows_symmetric_hessenberg()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1, 0, 1 };
            _matrix1[1] = new double[] { 0, 1, 2 };
            _matrix1[2] = new double[] { 1, 2, 0 };
        }

        /// <summary>
        /// Example from:
        /// http://web.csulb.edu/~tgao/math423/s93.pdf
        /// </summary>
        protected void InitData_dataset_4_rows_symmetric_hessenberg()
        {
            _matrix1 = new double[4][];
            _matrix1[0] = new double[] { 4, 1, -2, 2 };
            _matrix1[1] = new double[] { 1, 2, 0, 1 };
            _matrix1[2] = new double[] { -2, 0, 3,-2 };
            _matrix1[3] = new double[] { 2, 1, -2,-1 };

            //This is used in Multiply
            _matrix2 = new double[4][];
            _matrix2[0] = new double[] { 4, 1, -2, 2 };
            _matrix2[1] = new double[] { 1, 2, 0, 1 };
            _matrix2[2] = new double[] { -2, 0, 3, -2 };
            _matrix2[3] = new double[] { 2, 1, -2, -1 };
        }

        protected void InitData_dataset_3_rows_non_square()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1, 2 };
            _matrix1[1] = new double[] { 3, 4 };
            _matrix1[2] = new double[] { 5, 6 };
        }

        protected void InitData_dataset_nonsquare_matrix()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 1, 3, 5 };
            _matrix1[1] = new double[] { 2, 4,6 };
            
            _matrix2 = new double[3][];
            _matrix2[0] = new double[] { 7, 8 };
            _matrix2[1] = new double[] { 9, 10 };
            _matrix2[2] = new double[] { 11, 12 };
        }

        protected void InitData_dataset_invalid()
        {
            _matrix1 = new double[4][];
            _matrix1[0] = new double[] { 1, 2 };
            _matrix1[1] = new double[] { 3, 4 };
        }

        #endregion

       
    }
}
