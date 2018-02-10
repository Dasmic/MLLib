using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        [TestMethod]
        public void Matrix_reduced_3_row_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            
            PrivateObject obj = new PrivateObject(_mo);
            double[][] newMatrix = (double[][])obj.Invoke("GetReducedMatrix", new object[] { _matrix1,
                                         0,0 });

            Assert.AreEqual(newMatrix.Length,2);
            Assert.AreEqual(newMatrix[0].Length, 2); 

            Assert.AreEqual(newMatrix[0][0], 5);
            Assert.AreEqual(newMatrix[0][1], 6);
            Assert.AreEqual(newMatrix[1][0], 8);
            Assert.AreEqual(newMatrix[1][1], 9);
        }

        /// <summary>
        /// Example from:
        /// http://math.tutorvista.com/algebra/cofactor-matrix.html
        /// </summary>
        [TestMethod]
        public void Matrix_cofactor_2_row_test()
        {
            _matrix1 = new double[2][];
            _matrix1[0] = new double[] { 2, 4 };
            _matrix1[1] = new double[] { 5, 3 };

            double[][] newMatrix = _mo.CofactorMatrix(_matrix1);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], 3);
            Assert.AreEqual(newMatrix[0][1], -5);
            Assert.AreEqual(newMatrix[1][0], -4);
            Assert.AreEqual(newMatrix[1][1], 2);
        }


        /// <summary>
        /// Example from:
        /// http://www.mathwords.com/c/cofactor_matrix.htm
        /// </summary>
        [TestMethod]
        public void Matrix_cofactor_3_row_test()
        {
            _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 1, 0,1 };
            _matrix1[1] = new double[] { 2, 4,0 };
            _matrix1[2] = new double[] { 3, 5,6 };

            double[][] newMatrix = 
                _mo.CofactorMatrix(_matrix1);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 24);
            Assert.AreEqual(newMatrix[0][1], -12);
            Assert.AreEqual(newMatrix[0][2], -2);

            Assert.AreEqual(newMatrix[1][0], 5);
            Assert.AreEqual(newMatrix[1][1], 3);
            Assert.AreEqual(newMatrix[1][2], -5);

            Assert.AreEqual(newMatrix[2][0], -4);
            Assert.AreEqual(newMatrix[2][1], 2);
            Assert.AreEqual(newMatrix[2][2], 4);
        }

    }
}
