using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.Portable.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    [TestClass]
    public class PerceptronSingleUnitest:BaseTest
    {
        [TestMethod]
        public void NN_perceptron_single_sgd_single_training_sample_class_0()
        {
            initData_NN_dataset_linear_subgd_jason_example();
            BuildSingleUnitPerceptronSGD build = 
                new BuildSingleUnitPerceptronSGD();

            setPrivateVariablesInBuildObject(build);

            //Set params
            //build.setParameters(1,1,.45);            
            ModelSingleUnitPerceptron model =
                (ModelSingleUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }


        [TestMethod]
        public void NN_perceptron_single_sgd_single_training_sample_class_1()
        {
            initData_NN_dataset_linear_subgd_jason_example();
            BuildSingleUnitPerceptronSGD build = new BuildSingleUnitPerceptronSGD();

            setPrivateVariablesInBuildObject(build);

            //Set params
            //build.setParameters(1,1,.45);

            ModelSingleUnitPerceptron model =
                (ModelSingleUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void NN_perceptron_single_sgd_all_training_samples()
        {
            initData_NN_dataset_linear_subgd_jason_example();
            BuildSingleUnitPerceptronSGD build = new BuildSingleUnitPerceptronSGD();

            setPrivateVariablesInBuildObject(build);

            //Set params
            //build.setParameters(1,1,.45);

            
            ModelSingleUnitPerceptron model =
                (ModelSingleUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count=0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);
                if (SupportFunctions.DoubleCompare(value, _trainingData[_indexTargetAttribute][row]))
                    count++;
            }

            Assert.AreEqual(count,
                10);
        }


    }
}
