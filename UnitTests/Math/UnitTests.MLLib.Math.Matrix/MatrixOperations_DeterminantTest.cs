using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        [TestMethod]
        public void Matrix_determinant_triangular_2_row_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 2, 0 };
            _matrix1[1] = new double[] { 3, 3 };

            double det=
                _mo.DeterminantTriangular(_matrix1);

            //Check Det
            Assert.AreEqual(det, 6);            
        }


        /// <summary>
        /// Example taken from:
        /// 
        /// https://s-mat-pcs.oulu.fi/~mpa/matreng/eem3_4-3.htm
        /// </summary>
        [TestMethod]
        public void Matrix_determinant_3_row_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1,2, -2 };
            _matrix1[1] = new double[] { 2,0, 1 };
            _matrix1[2] = new double[] { 2,-1, 3 };

            double det =
                _mo.Determinant(_matrix1);

            //Check Det
            Assert.AreEqual(det, -3);
        }

        [TestMethod]
        public void Matrix_determinant_permutation_2_row_no_reorder_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 1, 0 };
            _matrix1[1] = new double[] { 0, 1 };

            double det =
                _mo.DeterminantPermutation(_matrix1);

            //Check Det
            Assert.AreEqual(det, 1);
        }

        [TestMethod]
        public void Matrix_determinant_permutation_2_row_reorder_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 0, 1 };
            _matrix1[1] = new double[] { 1, 0 };

            double det =
                _mo.DeterminantPermutation(_matrix1);

            //Check Det
            Assert.AreEqual(det, -1);
        }

        [TestMethod]
        public void Matrix_determinant_permutation_3_row_reorder_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 0, 0,1 };
            _matrix1[1] = new double[] { 0, 1,0 };
            _matrix1[2] = new double[] { 1, 0,0 };

            double det =
                _mo.DeterminantPermutation(_matrix1);

            //Check Det
            Assert.AreEqual(det, -1);
        }


        //All 3 lines exchanges
        [TestMethod]
        public void Matrix_determinant_permutation_3_row_reorder_all_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 0, 1, 0 };
            _matrix1[1] = new double[] { 0, 0, 1 };
            _matrix1[2] = new double[] { 1, 0, 0 };

            double det =
                _mo.DeterminantPermutation(_matrix1);

            //Check Det
            Assert.AreEqual(det, 1);
        }

    }

}
