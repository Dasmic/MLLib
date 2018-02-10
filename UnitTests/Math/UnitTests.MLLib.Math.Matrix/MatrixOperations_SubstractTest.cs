using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Substract
        [TestMethod]
        public void Matrix_substract_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.Substract(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], -4);
            Assert.AreEqual(newMatrix[0][1], -4);
            Assert.AreEqual(newMatrix[1][0], -4);
            Assert.AreEqual(newMatrix[1][1], -4);
        }


        /// <summary>
        /// Results:
        /// http://www.calcul.com/show/calculator/matrix-multiplication
        /// </summary>
        [TestMethod]
        public void Matrix_substract_nonsquare_row_test()
        {
            InitData_dataset_3_rows_non_square();
            double[][] newMatrix =
                _mo.Substract(_matrix1, _matrix1);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], 0);
            Assert.AreEqual(newMatrix[0][1], 0);
            
            Assert.AreEqual(newMatrix[1][0], 0);
            Assert.AreEqual(newMatrix[1][1], 0);
           
            Assert.AreEqual(newMatrix[2][0], 0);
            Assert.AreEqual(newMatrix[2][1], 0);
           
        }


        [TestMethod]
        public void Matrix_substract_nonsquare_single_column_test()
        {

            double[][] _matrix1 = new double[1][];
            _matrix1[0] = new double[] { 0, 1, 0 };
          
            double[][] _matrix2 = new double[1][];
            _matrix2[0] = new double[] { 1, 2, 3 };
            

            double[][] newMatrix =
                _mo.Substract(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 1);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], -1);
            Assert.AreEqual(newMatrix[0][1], -1);
            Assert.AreEqual(newMatrix[0][2], -3);
        }

        #endregion
    }
}
