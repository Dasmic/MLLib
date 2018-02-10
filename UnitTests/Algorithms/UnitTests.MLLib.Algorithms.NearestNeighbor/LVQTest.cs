using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.NearestNeighbour;
using Dasmic.MLLib.Common.MLCore;
    
namespace UnitTests.MLLib.Algorithms.NearestNeighbor
{
    [TestClass]
    public class LVQTest:BaseTest
    {
        [TestMethod]
        public void LVQ_jason_single_training_sample_positive()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildLVQ lvq = new BuildLVQ();
            
            ModelLVQ model = 
                (ModelLVQ)lvq.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            double [] data = GetSingleTrainingRowDataForTest(0);
            double value= model.RunModelForSingleData(data);

            Assert.AreEqual(value, _trainingData[_indexTargetAttribute][0]);
        }

        [TestMethod]
        public void LVQ_jason_single_training_sample_negative()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildLVQ lvq = new BuildLVQ();
            lvq.SetParameters(3.4, 5);
            ModelLVQ model =
                (ModelLVQ)lvq.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);

            
            double[] data = GetSingleTrainingRowDataForTest(4);
            double value = model.RunModelForSingleData(data);

            Assert.AreNotEqual(value, _trainingData[_indexTargetAttribute][4]);
        }

        [TestMethod]
        public void LVQ_jason_all_training_sample()
        {
            initData_dataset_gaussian_kNN_jason_example();
            BuildLVQ lvq = new BuildLVQ();
            
            ModelLVQ model = (ModelLVQ) lvq.BuildModel(_trainingData, _attributeHeaders, _indexTargetAttribute);
            int count=0;
            for(int row=0;row<_trainingData[0].Length;row++)
            {                
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                if (value == _trainingData[_indexTargetAttribute][row])
                    count++;
            }

            Assert.AreEqual(count, 10);
        }

    }
}
