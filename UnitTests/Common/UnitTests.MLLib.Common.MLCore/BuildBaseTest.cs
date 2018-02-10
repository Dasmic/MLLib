using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.MLCore;

namespace UnitTests.MLLib.Common.MLCore
{
    [TestClass]
    public class BuildBaseTest:Base
    {
        [TestMethod]
        public void BuildBase_unique_values()
        {
            initData_dataset_7_row_2_class_example();

            BuildBaseObject bbo =
                    new BuildBaseObject();

            PrivateObject obj = new PrivateObject(bbo);
            HashSet<double> meanMatrix = (HashSet<double>)obj.Invoke(
                                "GetUniqueValues",
                                new object[] { _trainingData[_indexTargetAttribute]});

            Assert.AreEqual(meanMatrix.Count,2);
            Assert.IsTrue(meanMatrix.Contains(1));
            Assert.IsTrue(meanMatrix.Contains(2));
        }

        [TestMethod]
        public void BuildBase_class_based_matrix_example()
        {
            initData_dataset_7_row_2_class_example();

            BuildBaseObject bbo =
                    new BuildBaseObject();
            setPrivateVariablesInBuildObject(bbo);

            //Set values of private members            
            PrivateObject obj = new PrivateObject(bbo);

            //Now call Functions
            HashSet<double> uniqueValues =
                (HashSet<double>)obj.Invoke(
                               "GetUniqueValues",
                               new object[] { _trainingData[_indexTargetAttribute] });

            double[] classValues = new double[uniqueValues.Count];
            List<double[][]> _classMatrix =
                  (List<double[][]>)obj.Invoke(
                               "GetClassBasedInputMatrix",
                               new object[] { uniqueValues,
                                         classValues });

            Assert.AreEqual(_classMatrix.Count, 2);

            Assert.AreEqual(_classMatrix[0][0].Length, 4);
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[0][0][0], _classMatrix1[0][0]));
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[0][0][1], _classMatrix1[0][1]));
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[0][0][2], _classMatrix1[0][2]));
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[0][0][3], _classMatrix1[0][3]));

            Assert.AreEqual(_classMatrix[1][0].Length, 3);
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[1][0][0], _classMatrix2[0][0]));
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[1][0][1], _classMatrix2[0][1]));
            Assert.IsTrue(SupportFunctions.DoubleCompare(_classMatrix[1][0][2], _classMatrix2[0][2])); 
        }


        [TestMethod]
        public void BuildBase_class_mean_2_row_data()
        {
            initData_classdata_4_rows_2_class();
            BuildBaseObject bbo =
                     new BuildBaseObject();
            setPrivateVariablesInBuildObject(bbo);
            
            PrivateObject obj = new PrivateObject(bbo);
            List<double[][]> allMeans = (List<double[][]>)obj.Invoke("GetClassMeanMatrix",
                                            new object[] { _classInputMatrix });

            Assert.IsTrue(allMeans[0][0][0] == 80);
            Assert.IsTrue(allMeans[0][1][0] == 140);
            Assert.IsTrue(allMeans[1][0][0] == 40);
            Assert.IsTrue(allMeans[1][1][0] == 75);
        }

        [TestMethod]
        public void BuildBase_class_sd_2_row_data()
        {
            initData_classdata_4_rows_2_class();
            BuildBaseObject bbo =
                     new BuildBaseObject();
            setPrivateVariablesInBuildObject(bbo);

            PrivateObject obj = new PrivateObject(bbo);
            List<double[][]> allMeans = (List<double[][]>)obj.Invoke("GetClassMeanMatrix",
                                            new object[] { _classInputMatrix });
            List<double[][]> allSD = (List<double[][]>)obj.Invoke("GetClassStandardDeviationMatrix",
                                            new object[] { _classInputMatrix,allMeans });

            Assert.IsTrue(SupportFunctions.DoubleCompare(allSD[0][0][0],System.Math.Sqrt(400)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(allSD[0][1][0], System.Math.Sqrt(3600)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(allSD[1][0][0], System.Math.Sqrt(100)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(allSD[1][1][0], System.Math.Sqrt(289)));
        }


    }
}
