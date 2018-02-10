using Dasmic.MLLib.Algorithms.SupportVectorMachine;
using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.SupportVectorMachine
{
    [TestClass]
    public class SVMLinearSubGDTest:BaseTest
    {    
        [TestMethod]
        public void SVM_linear_subgd_single_training_sample()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMLinearSubGD build = new BuildSVMLinearSubGD();

            setPrivateVariablesInBuildObject(build);

            //Set params
            build.SetParameters(1,1,.45);

            ModelSVMLinearSubGD model =
                (ModelSVMLinearSubGD)build.BuildModel(
                    _trainingData, _attributeHeaders, 
                    _indexTargetAttribute);

            double[] data = GetSingleTrainingRowDataForTest(0);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value, 
                _trainingData[_indexTargetAttribute][0]);
        }

        [TestMethod]
        public void SVM_linear_subgd_all_training_samples()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMLinearSubGD build = new BuildSVMLinearSubGD();

            setPrivateVariablesInBuildObject(build);
            //Set params
            build.SetParameters(1, 1, .45);

            ModelSVMLinearSubGD model =
                (ModelSVMLinearSubGD)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            for(int row=0;row<_trainingData[0].Length;row++)
            { 
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                Assert.AreEqual(value,
                    _trainingData[_indexTargetAttribute][row]);
            }
        }
    }
}
