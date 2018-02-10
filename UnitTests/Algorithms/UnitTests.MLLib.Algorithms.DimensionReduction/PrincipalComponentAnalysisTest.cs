using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DimensionReduction;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.DimensionReduction
{
    [TestClass]
    public class PrincipalComponentAnalysisTest:BaseTest
    {
        #region 2 Row Tests

        [TestMethod]
        public void PCA_new_matrix_2_rows()
        {
            InitData_dataset_pca_example();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            double[][] rotationMatrix=null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData, 
                _attributeHeaders,ref standardDeviation,ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], .82));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], -1.77));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], .99));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][3], .27));

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][0], -.17));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][1], .14));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][2], .38));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[1][3], .13));
         }


        [TestMethod]
        public void PCA_new_matrix_2_rows_rank_1()
        {
            InitData_dataset_pca_example();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 1;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 1);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][0], .82));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][1], -1.77));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][2], .99));
            Assert.IsTrue(SupportFunctions.DoubleCompare(newMatrix[0][3], .27));            
        }

        /// <summary>
        /// R:
        /// prcomp(A)
        /// </summary>
        [TestMethod]
        public void PCA_new_matrix_3_rows_rank_3_0()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 3;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 1.26));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[1], .85));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .29));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .60));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], -.74));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][0], -.52));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][1], .75));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][2], .40));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][0], .80));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][1], .26));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][2], .53));
        }


        [TestMethod]
        public void PCA_new_matrix_3_rows_rank_2_0()
        {
            InitData_dataset_3_rows_symmetric_hessenberg();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 2;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(standardDeviation.Length, 2);
            Assert.AreEqual(rotationMatrix.Length, 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 1.26));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[1], .85));
            

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .29));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .60));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], -.74));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][0], -.52));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][1], .75));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][2], .40));
        }

        /// <summary>
        /// R:
        /// prcomp(A)
        /// </summary>
        [TestMethod]
        public void PCA_new_matrix_3_rows_rank_3_1()
        {
            InitData_dataset_3_rows_non_symmetric();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 3;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 1.73));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[1], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[2], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .57));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .57));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], .57));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][0], -.81));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][1], .40));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][2], .40));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][0], 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][1], -.70));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][2], .70));            
        }

        /// <summary>
        /// R:
        /// prcomp(A)
        /// </summary>
        [TestMethod]
        public void PCA_new_matrix_3_rows_tol_3_0()
        {
            InitData_dataset_3_rows_non_symmetric();
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Tolerance = .1;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 1);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 1.73));
            

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .57));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .57));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], .57));
        }

        /// <summary>
        /// R:
        /// prcomp(A)
        /// </summary>
        [TestMethod]
        public void PCA_new_matrix_4_rows_rank_4_0()
        {
            InitData_dataset_4_rows_symmetric_hessenberg(); ;
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 4;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 4);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 3.70));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[1], 1.27));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[2], .689));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[3], 0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .67));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], -.60));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][3], .40));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][0], -.05));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][1], .24));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][2], .52));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][3], .81));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][0], .40));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][1], -.85));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][2], .31));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[2][3], .07));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[3][0], .61));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[3][1], .44));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[3][2], .50));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[3][3], -.41));
        }


        /// <summary>
        /// R:
        /// prcomp(A)
        /// </summary>
        [TestMethod]
        public void PCA_new_matrix_4_rows_rank_2_0()
        {
            InitData_dataset_4_rows_symmetric_hessenberg(); ;
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis();

            pca.Rank = 2;
            double[][] rotationMatrix = null;
            double[] standardDeviation = null;
            double[][] newMatrix = pca.GetPrincipleFeatures(_trainingData,
                _attributeHeaders, ref standardDeviation, ref rotationMatrix);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(standardDeviation.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, _trainingData[0].Length);

            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[0], 3.70));
            Assert.IsTrue(SupportFunctions.DoubleCompare(standardDeviation[1], 1.27));            

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][0], .67));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][1], .12));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][2], -.60));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[0][3], .40));

            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][0], -.05));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][1], .24));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][2], .52));
            Assert.IsTrue(SupportFunctions.DoubleCompare(rotationMatrix[1][3], .81));
         
        }

        #endregion
    }
}
