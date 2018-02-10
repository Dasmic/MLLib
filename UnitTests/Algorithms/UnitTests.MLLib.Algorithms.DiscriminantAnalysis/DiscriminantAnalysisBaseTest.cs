using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DiscriminantAnalysis;
using Dasmic.Portable.Core;



namespace UnitTests_DiscriminantAnalysis
{
    [TestClass]
    public class DiscriminantAnalysisBaseTest:BaseTest
    {
        
        public DiscriminantAnalysisBaseTest()
        {

        }

        #region Initialization
        private void initData_dataset_2_rows()
        {
            _attributeHeaders = new string[] {
                                     "Attr 1",
                                    "Attr 2",
                                    "Value"};

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 100, 60 };
            _trainingData[1] = new double[] { 200, 80 };
            _trainingData[2] = new double[] { 1, 0 };

            _indexTargetAttribute = 2;
        }
        

        private void initData_classdata_4_rows_2_class()
        {
            _classInputMatrix =
                new List<double[][]>();


            _attributeHeaders = new string[] {
                                     "Attr 1",
                                    "Attr 2",
                                    "Value"};

            _trainingData = new double[3][];
            _trainingData[0] = new double[] { 100, 60,30,50 };
            _trainingData[1] = new double[] { 200, 80,92,58 };
            _trainingData[2] = new double[] { 1, 0,1,0 };
            _indexTargetAttribute = 2;

            _classMatrix1 = new double[2][];
            _classMatrix1[0] = new double[] { 100, 60 };
            _classMatrix1[1] = new double[] { 200, 80 };

            _classMatrix2 = new double[2][];
            _classMatrix2[0] = new double[] { 30, 50 };
            _classMatrix2[1] = new double[] { 92, 58 };

            _classInputMatrix.Add(_classMatrix1);
            _classInputMatrix.Add(_classMatrix2);

            _indexTargetAttribute = 2;
        }
        #endregion

        #region Dataset Unit Tests

        [TestMethod]
        public void DA_dataset_mean_2_row_data()
        {
            initData_dataset_2_rows();
            BuildBase lda =
                  getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] means = (double[])obj.Invoke("getDataSetMeanMatrix",
                                new object[] {});

            Assert.IsTrue(means[0] == 80);
            Assert.IsTrue(means[1] == 140);
        }

        [TestMethod]
        public void DA_dataset_mean_7_row_data_example()
        {
            initData_dataset_7_row_2_class_example();
            BuildBase lda =
                  getBuildAndAssignValues();

            
            PrivateObject obj = new PrivateObject(lda);
            double[] means = (double[])obj.Invoke("getDataSetMeanMatrix",
                                new object[] {  });

            Assert.IsTrue(means[0] >= 2.87 && means[0] <= 2.89);
            Assert.IsTrue(means[1] >= 5.66 && means[0] <= 5.69);
            
        }

#endregion

       
        [TestMethod]
        public void DA_corrected_matrix_2_row_data()
        {
            initData_classdata_4_rows_2_class();

            //double[] meanMatrix = 
            //    new double[] { 60, 107.5 };

            BuildBase lda =
                    getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] meanMatrix = (double[])obj.Invoke(
                                "getDataSetMeanMatrix",
                                new object[] { });
            
            
            List<double[][]> correctedData = (List<double[][]>)obj.Invoke("getCorrectedDataMatrix",
                                            new object[] { _classInputMatrix, meanMatrix });

            double[][] tmp = correctedData[0];
            Assert.IsTrue(tmp[0][0] == 40);
            Assert.IsTrue(tmp[0][1] == 0);
            Assert.IsTrue(tmp[1][0] == 92.5);
            Assert.IsTrue(tmp[1][1] == -27.5);
            
            tmp = correctedData[1];
            Assert.IsTrue(tmp[0][0] == -30);
            Assert.IsTrue(tmp[0][1] == -10);
            Assert.IsTrue(tmp[1][0] == -15.5);
            Assert.IsTrue(tmp[1][1] == -49.5);
        }

        [TestMethod]
        public void DA_corrected_matrix_example_data()
           {
            initData_dataset_7_row_2_class_example();
          
            
            List<double[][]> allData = new List<double[][]>();
            allData.Add(_classMatrix1);
            allData.Add(_classMatrix2);

            BuildBase lda =
                    getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] meanMatrix = (double[])obj.Invoke(
                               "getDataSetMeanMatrix",
                               new object[] { });

            List<double[][]> correctedData = (List<double[][]>)obj.Invoke("getCorrectedDataMatrix",
                                            new object[] { allData,
                                                meanMatrix });

            double[][] tmp = correctedData[0];
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][0],0.060));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][1], -.357));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][2], 0.679));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][3], 0.269));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][0], 0.951));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][1], 2.109));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][2], -0.025));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][3], -.209));

            tmp = correctedData[1];
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][0], -.305));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][1], -0.732));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][2], 0.386));            
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][0], -1.218));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][1], 0.547));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][2], -2.155));
        }
        

        [TestMethod]
        public void DA_covariance_matrix_example_data()
        {
            initData_dataset_7_row_2_class_example();


            List<double[][]> allData = new List<double[][]>();
            allData.Add(_classMatrix1);
            allData.Add(_classMatrix2);

            BuildBase lda =
                    getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] meanMatrix = (double[])obj.Invoke(
                               "getDataSetMeanMatrix",
                               new object[] {});

            List<double[][]> correctedData = (List<double[][]>)obj.Invoke("getCorrectedDataMatrix",
                                            new object[] { allData,
                                                meanMatrix });

            List<double[][]> covMatrix = (List<double[][]>)obj.Invoke("getCoVarianceMatrix",
                                            new object[] { correctedData });

            double[][] tmp = covMatrix[0];
            Assert.AreEqual(tmp.Length,2);
            Assert.AreEqual(tmp[0].Length, 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][0], 0.166));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][1], -.192));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][0], -.192));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][1], 1.349));
            

            tmp = covMatrix[1];
            Assert.AreEqual(tmp.Length, 2);
            Assert.AreEqual(tmp[0].Length, 2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][0], 0.259));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[0][1], -0.286));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][0], -0.286));
            Assert.IsTrue(SupportFunctions.DoubleCompare(tmp[1][1], 2.142));
            
        }



        [TestMethod]
        public void DA_probability_class_matrix_example()
        {
            initData_dataset_7_row_2_class_example();
            
            List<double[][]> allData = new List<double[][]>();
            allData.Add(_classMatrix1);
            allData.Add(_classMatrix2);

            BuildBase lda =
                  getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] probabilityMatrix = (double[])obj.Invoke(
                               "getClassProbabilities",
                               new object[] { allData});
            
            Assert.AreEqual(probabilityMatrix.Length, 2);
            
            Assert.IsTrue(SupportFunctions.DoubleCompare(probabilityMatrix[0], (4.0/7.0)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(probabilityMatrix[1], (3.0/7.0)));
           
        }

        [TestMethod]
        public void DA_pooled_covariance_matrix_example()
        {
            initData_dataset_7_row_2_class_example();

            List<double[][]> allData = new List<double[][]>();
            allData.Add(_classMatrix1);
            allData.Add(_classMatrix2);

            BuildBase lda =
                  getBuildAndAssignValues();

            PrivateObject obj = new PrivateObject(lda);
            double[] probabilityMatrix = (double[])obj.Invoke(
                               "getClassProbabilities",
                               new object[] { allData});

            double[] meanMatrix = (double[])obj.Invoke(
                               "getDataSetMeanMatrix",
                               new object[] {  });

            List<double[][]> correctedData = (List<double[][]>)obj.Invoke("getCorrectedDataMatrix",
                                            new object[] { allData,
                                                meanMatrix });

            List<double[][]> covMatrix = (List<double[][]>)obj.Invoke("getCoVarianceMatrix",
                                            new object[] { correctedData });

            double[][] pooledCoVarMatrix = (double[][])obj.Invoke(
                               "getPooledCoVarianceMatrix",
                               new object[] { covMatrix,
                                   probabilityMatrix});

            Assert.AreEqual(pooledCoVarMatrix.Length, 2);
            Assert.AreEqual(pooledCoVarMatrix[0].Length, 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(pooledCoVarMatrix[0][0], 0.206));
            Assert.IsTrue(SupportFunctions.DoubleCompare(pooledCoVarMatrix[0][1], -0.233));
            Assert.IsTrue(SupportFunctions.DoubleCompare(pooledCoVarMatrix[1][0], -0.233));
            Assert.IsTrue(SupportFunctions.DoubleCompare(pooledCoVarMatrix[1][1], 1.689));    
        }
    }
}
