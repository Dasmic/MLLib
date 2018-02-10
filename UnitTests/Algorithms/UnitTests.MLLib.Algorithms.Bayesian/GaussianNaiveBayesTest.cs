using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.Bayesian;

namespace UnitTests.MLLib.Algorithms.Bayesian
{
    [TestClass]
    public class GaussianNaiveBayesTest:BaseTest
    {
        [TestMethod]
        public void NaiveBayes_gaussian_jason_single_training_sample_positive()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();

            BuildGaussianNaiveBayes bnb = new BuildGaussianNaiveBayes();

            ModelBase model = (ModelBase)bnb.BuildModel
                                (_trainingData, _attributeHeaders, _indexTargetAttribute);

            double[] data;
            double value;
            data = GetSingleTrainingRowDataForTest(0);

            value = model.RunModelForSingleData(data);
            Assert.AreEqual(value, 
                _trainingData[_indexTargetAttribute][0]);
        }

        [TestMethod]
        public void NaiveBayes_gaussian_jason_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();

            BuildGaussianNaiveBayes bnb = new BuildGaussianNaiveBayes();
            ModelBase model = (ModelBase)bnb.BuildModel
                                (_trainingData, _attributeHeaders, _indexTargetAttribute);

            double[] data;
            double value;
            int count = 0;
            data = new double[_trainingData.Length - 1];
            for(int row=0;row<_trainingData[0].Length;row++)
            { 
                for (int col = 0; col < _trainingData.Length - 1; col++)
                {
                    data[col] =
                         _trainingData[col][row];
                }
                value = model.RunModelForSingleData(data);
                if (value == _trainingData[_indexTargetAttribute][row]) count++;
            }
            
            Assert.AreEqual(count,10);
        }
    }
}
