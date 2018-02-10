using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.NearestNeighbour;
using Dasmic.MLLib.Common.MLCore;
    
namespace UnitTests.MLLib.Algorithms.NearestNeighbor
{
    [TestClass]
    public class kNNTest:BaseTest
    {
        [TestMethod]
        public void kNN_jason_single_training_sample()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildkNN kNN = new BuildkNN();
            ModelkNN model = 
                (ModelkNN)kNN.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            double [] data = GetSingleTrainingRowDataForTest(0);
            double value= model.RunModelForSingleData(data);

            Assert.AreEqual(value, _trainingData[_indexTargetAttribute][0]);
        }

        [TestMethod]
        public void kNN_jason_all_training_sample()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildkNN kNN = new BuildkNN();
            ModelkNN model = (ModelkNN) kNN.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            for(int row=0;row<_trainingData[0].Length;row++)
            {
                model.setParameters(null, 5); //with k of 3 doesnt work
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
            }
        }

        [TestMethod]
        public void kNN_jason_given_validation_sample()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildkNN kNN = new BuildkNN();
            ModelkNN model = (ModelkNN)kNN.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            //model.setParameters(null, 5); //with k of 3 doesnt work
            double[] data = { 8.093607318, 3.365731514 };
            double value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 1);

        }
    }
}
