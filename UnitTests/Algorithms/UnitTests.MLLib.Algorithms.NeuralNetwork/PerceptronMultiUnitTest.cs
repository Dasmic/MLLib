using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.Portable.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    [TestClass]
    public class PerceptronMultiUnitTest:BaseTest
    {
        #region Training Data from Jason Gaussian Naive Bayes
        [TestMethod]
        public void NN_perceptron_multi_single_training_sample_gnb_class_0()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildMultiUnitPerceptronSGD build = 
                    new BuildMultiUnitPerceptronSGD();
           
            ModelMultiUnitPerceptron model =
                (ModelMultiUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void NN_perceptron_multi_single_training_sample_gnb_class_1()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildMultiUnitPerceptronSGD build =
                    new BuildMultiUnitPerceptronSGD();

            ModelMultiUnitPerceptron model =
                (ModelMultiUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            int row = 5;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }


        [TestMethod]
        public void NN_perceptron_multi_all_training_samples_gnb()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildMultiUnitPerceptronSGD build =
                    new BuildMultiUnitPerceptronSGD();

            ModelMultiUnitPerceptron model =
                (ModelMultiUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < 10; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (SupportFunctions.DoubleCompare(value, 
                        _trainingData[_indexTargetAttribute][row]))
                    count++;
            }

            Assert.AreEqual(count,
                10);
        }
        #endregion

        #region Training Data from Jason  Naive Bayes

        [TestMethod]
        public void NN_perceptron_multi_single_training_sample_nb_class_1()
        {
            initData_dataset_naive_bayes_jason_example();
            BuildMultiUnitPerceptronSGD build =
                    new BuildMultiUnitPerceptronSGD();

            ModelMultiUnitPerceptron model =
                (ModelMultiUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void NN_perceptron_multi_all_training_samples_nb()
        {
            initData_dataset_naive_bayes_jason_example();
            BuildMultiUnitPerceptronSGD build =
                    new BuildMultiUnitPerceptronSGD();

            build.SetParameters(.3, 1000);
            ModelMultiUnitPerceptron model =
                (ModelMultiUnitPerceptron)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < 10; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (SupportFunctions.DoubleCompare(value,
                        _trainingData[_indexTargetAttribute][row]))
                    count++;
            }

            Assert.AreEqual(count,
                5);
        }

        #endregion

    }
}
