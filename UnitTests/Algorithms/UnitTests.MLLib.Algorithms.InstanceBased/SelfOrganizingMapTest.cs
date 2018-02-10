using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.InstanceBased;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.DataManagement;

namespace UnitTests.MLLib.Algorithms.InstanceBased
{
    #region Value Compare
    [TestClass]
    public class SelfOrganizingMapTest:BaseTest
    {
        /// <summary>
        /// R:
        /// library(som)
        /// obj<-som.init(A, 2, 2, init="linear")
        /// </summary>
        [TestMethod]
        public void SOM_2_columns_2_category_0()
         {
            InitData_dataset_2_rows_2_category();
            BuildSelfOrganizingMap build = new BuildSelfOrganizingMap();
            //build.SetParameters(2, 2);//, .9, 100);//,0, 4);
            ModelSelfOrganizingMap model =
                    (ModelSelfOrganizingMap) build.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            //Compare Weights          
            String somPrint = model.GetPrintedSOMMap().ToString();
            Assert.AreEqual(somPrint, "1 0 \r\n0.39 0.61 \r\n0.39 0.61 \r\n0 1 \r\n");

            //All weights should be same
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(0), 1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(1), 0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][1].GetWeight(0), 0.39));
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][1].GetWeight(1), 0.61));
        }

        /// <summary>
        /// R:
        /// library(som)
        /// obj<-som.init(data, 2, 2, init="linear")
        /// </summary>
        [TestMethod]
        public void SOM_2_columns_1_category_0()
        {
            InitData_dataset_2_rows_1_category();
            BuildSelfOrganizingMap build = new BuildSelfOrganizingMap();
            build.SetParameters(2, 2, .9, 100);
            ModelSelfOrganizingMap model =
                    (ModelSelfOrganizingMap)build.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            //Compare Weights
            String somPrint = model.GetPrintedSOMMap().ToString();
            Assert.AreEqual(somPrint, "1 1 \r\n1 1 \r\n1 1 \r\n1 1 \r\n");
            
            //All weights should be same
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(0), model.SomMap[0][0].GetWeight(1))) ;
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(0), model.SomMap[1][0].GetWeight(1)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(0), model.SomMap[0][1].GetWeight(1)));
            Assert.IsTrue(SupportFunctions.DoubleCompare(model.SomMap[0][0].GetWeight(0), model.SomMap[1][1].GetWeight(1)));                       
        }


        /// <summary>
        /// R:
        /// library(som)
        /// obj<-som.init(data, 2, 2, init="linear")
        /// </summary>
        [TestMethod]
        public void SOM_6_columns_iris_0()
        {
            InitData_dataset_iris();
            
            //Split last column away from _trainingData
            double[][] newTrainingData = ArrayManipulation.RemoveLastColumn2D(_trainingData);
            string [] values = Conversion.ConvertToStringArray(ArrayManipulation.GetLastColAs1DArray(_trainingData));
            string[] newAttributeHeaders = ArrayManipulation.RemoveLastColumn1D(_attributeHeaders);           

            BuildSelfOrganizingMap build = new BuildSelfOrganizingMap();
            build.SetParameters(4, 4);//, .1, 100, 3);
            ModelSelfOrganizingMap model =
                    (ModelSelfOrganizingMap)build.BuildModel(newTrainingData, newAttributeHeaders);

            values = Conversion.ReplaceWithString1D(values, "0", "setosa");
            values = Conversion.ReplaceWithString1D(values, "1", "versicolor");
            values = Conversion.ReplaceWithString1D(values, "2", "virginica");

            string somMap = model.GetPrintedSOMMapWithNameIds(newTrainingData,
                                    values);
            //string valueCompare = "versicolor versicolor setosa setosa \r\n" +
            //                     "virginica versicolor versicolor setosa \r\n" +
            //                     "virginica virginica virginica virginica \r\n" +
            //                     "virginica virginica virginica virginica";

            //Using Random init weights is leading to different structure
            //However the following line should always be there
            Assert.IsTrue(somMap.Contains("virginica virginica virginica virginica"));
        }
        #endregion Value compare

        #region Name Id
        /// <summary>
        /// R:
        /// library(som)
        /// obj<-som.init(data, 2, 2, init="linear")
        /// </summary>
        [TestMethod]
        public void SOM_2_columns_1_category_0_name_id()
        {
            InitData_dataset_2_rows_1_category();
            BuildSelfOrganizingMap build = new BuildSelfOrganizingMap();
            build.SetParameters(2, 2, .1, 100, 3);
            ModelSelfOrganizingMap model =
                    (ModelSelfOrganizingMap)build.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            string somMap = model.GetPrintedSOMMapWithNameIds(_trainingData, 
                                    new string[] { "A", "B"});
                              
            Assert.AreEqual(somMap, "B B \r\nB B \r\n");            
        }

        /// <summary>
        /// R:
        /// library(som)
        /// obj<-som.init(data, 2, 2, init="linear")
        /// </summary>
        [TestMethod]
        public void SOM_2_columns_2_category_0_name_id()
        {
            InitData_dataset_2_rows_2_category();
            BuildSelfOrganizingMap build = new BuildSelfOrganizingMap();
            //build.SetParameters(2, 2), .1, 100, 3);
            ModelSelfOrganizingMap model =
                    (ModelSelfOrganizingMap)build.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            string somMap = model.GetPrintedSOMMapWithNameIds(_trainingData,
                                    new string[] { "A", "B" });

            Assert.IsTrue(somMap.Contains("B B"));
        }

#endregion

    }
}
