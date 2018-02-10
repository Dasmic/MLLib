using Dasmic.MLLib.Algorithms.EnsembleCART;
using Dasmic.MLLib.Common.MLCore;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.EnsembleCART
{
    [TestClass]
    public class AdaBoostTest:BaseTest
    {
        [TestMethod]
        public void AdaBoost_single_training_sample_value_0()
        {
            initData_Jason_AdaBoost();
            BuildAdaBoost build =
                    new BuildAdaBoost();

            ModelAdaBoost model =
                (ModelAdaBoost)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

             Assert.AreEqual(value, _trainingData[_indexTargetAttribute][row]);
        }


        [TestMethod]
        public void AdaBoost_single_training_sample_value_1()
        {
            initData_Jason_AdaBoost();
            BuildAdaBoost build =
                    new BuildAdaBoost();

            ModelAdaBoost model =
                (ModelAdaBoost)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            value = value > 0 ? 1 : 0;

            Assert.AreEqual(value, 
                    _trainingData[_indexTargetAttribute][row]);
        }


        [TestMethod]
        public void AdaBoost_all_training_samples()
        {
            initData_Jason_AdaBoost();
            BuildAdaBoost build =
                    new BuildAdaBoost();

            ModelAdaBoost model =
                (ModelAdaBoost)build.BuildModel(
                    _trainingData,
                    _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                value = value > 0 ? 1 : 0;
                if (value ==
                    _trainingData[_indexTargetAttribute][row])
                    count++;
            }
            Assert.AreEqual(count,9);
        }
    }
}
