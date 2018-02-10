using Dasmic.MLLib.Algorithms.EnsembleCART;
using Dasmic.MLLib.Common.MLCore;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.EnsembleCART
{
    [TestClass]
    public class RandomForestTest:BaseTest
    {
        [TestMethod]
        public void RandomForest_single_training_sample_value_0()
        {
            initData_Jason_Bagging();
            BuildRandomForest build =
                    new BuildRandomForest();
            
            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row=0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            
            Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
        }

       [TestMethod]
        public void RandomForest_single_training_sample_value_1()
        {
            initData_Jason_Bagging();
            BuildRandomForest build =
                    new BuildRandomForest();

            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            //Can be both 1 or 0
            //Assert.IsTrue(value == 0 | value == 1);
            Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
        }
        
        /// <summary>
        /// This test is failing sometimes
        /// </summary>
        [TestMethod]
        public void RandomForest_single_training_sample_border_value_0()
        {
            initData_Jason_Bagging();
            BuildRandomForest build =
                    new BuildRandomForest();
            
            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 4;//5.38 2.1 0
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            //Can be both 1 or 0
            //Assert.IsTrue(value==0 | value==1);
            Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
        }

        
        
        [TestMethod]
        public void RandomForest_all_training_samples()
        {
            initData_Jason_Bagging();
            BuildRandomForest build =
                    new BuildRandomForest();

            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

         
            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                if (value ==
                    _trainingData[_indexTargetAttribute][row])
                    count++;
            }
            
            Assert.AreEqual(9,count);
        }

        [TestMethod]
        public void RandomForest_single_training_sample_3_features()
        {
            initData_Jason_3_features();
            BuildRandomForest build =
                    new BuildRandomForest();
          
            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 4;//5.38 2.1 0
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);
            //Can be both 1 or 0
            //Assert.IsTrue(value==0 | value==1);
            Assert.AreEqual(value, 
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void RandomForest_all_training_samples_3_features()
        {
            initData_Jason_3_features();
            BuildRandomForest build =
                    new BuildRandomForest();

            ModelRandomForest model =
                (ModelRandomForest)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);


            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                if (value ==
                    _trainingData[_indexTargetAttribute][row])
                    count++;
            }

            Assert.AreEqual(9, count);
        }
    }
}
