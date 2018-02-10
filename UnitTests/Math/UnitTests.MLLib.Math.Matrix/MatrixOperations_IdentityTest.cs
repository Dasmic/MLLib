using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        [TestMethod]
        public void Matrix_identity_2_row_test()
        {

            double[][] iM =
                _mo.IdentityMatrix(2);

            Assert.AreEqual(iM.Length, 2);

            //Check Det
            Assert.AreEqual(iM[0][0], 1);
            Assert.AreEqual(iM[1][1], 1);
            Assert.AreEqual(iM[0][1], 0);
            Assert.AreEqual(iM[1][0], 0);
        }

        [TestMethod]
        public void Matrix_identity_3_row_test()
        {

            double[][] iM =
                _mo.IdentityMatrix(3);

            Assert.AreEqual(iM.Length, 3);

            //Check Det
            Assert.AreEqual(iM[0][0], 1);
            Assert.AreEqual(iM[1][1], 1);
            Assert.AreEqual(iM[2][2], 1);
            Assert.AreEqual(iM[0][1], 0);
            Assert.AreEqual(iM[0][2], 0);
        }
    }
}
