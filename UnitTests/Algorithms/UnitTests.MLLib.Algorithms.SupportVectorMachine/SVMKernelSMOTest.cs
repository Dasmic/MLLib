using Dasmic.MLLib.Algorithms.SupportVectorMachine;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace UnitTests.MLLib.Algorithms.SupportVectorMachine
{
    [TestClass]
    public class SVMKernelSMOTest:BaseTest
    {    
        [TestMethod]
        public void SVM_kernel_smo_single_training_sample()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMKernelSMO build = new BuildSVMKernelSMO();

            setPrivateVariablesInBuildObject(build);

            //Set params
            //build.setParameters(1,1,.45);

            ModelSVMKernelSMO model =
                (ModelSVMKernelSMO)build.BuildModel(
                    _trainingData, _attributeHeaders, 
                    _indexTargetAttribute);

            double[] data = GetSingleTrainingRowDataForTest(0);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value, 
                _trainingData[_indexTargetAttribute][0]);
        }

        
        [TestMethod]
        public void SVM_kernel_linear_smo_all_training_samples()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMKernelSMO build = new BuildSVMKernelSMO();

            setPrivateVariablesInBuildObject(build);
            //Set params
            build.SetParameters(2, .0001, .001);

            ModelSVMKernelSMO model =
                (ModelSVMKernelSMO)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count=0;
            for (int row=0;row<_trainingData[0].Length;row++)
            { 
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (value ==_trainingData[_indexTargetAttribute][row])
                    count++;      
            }

            //Random function is adding some randomness
            Assert.IsTrue(count >= 8 && count <=9); //80-90% accuracy
        }

        [TestMethod]
        public void SVM_kernel_poly_degree_3_smo_all_training_samples()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMKernelSMO build = new BuildSVMKernelSMO();

            setPrivateVariablesInBuildObject(build);
            //Set params
            build.SetParameters(2, .0001, .001);
            build.setKernel(BuildSVMKernelSMO.Kernels.Polynomial, 3);
            ModelSVMKernelSMO model =
                (ModelSVMKernelSMO)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (value == _trainingData[_indexTargetAttribute][row])
                    count++;
            }
            //Random function is adding some randomness
            Assert.IsTrue(count >= 9 && count <= 10);
        }


        [TestMethod]
        public void SVM_kernel_poly_degree_1_smo_all_training_samples()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMKernelSMO build = new BuildSVMKernelSMO();

            setPrivateVariablesInBuildObject(build);
            //Set params
            build.SetParameters(2, .0001, .001);
            build.setKernel(BuildSVMKernelSMO.Kernels.Polynomial, 1);
            ModelSVMKernelSMO model =
                (ModelSVMKernelSMO)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (value == _trainingData[_indexTargetAttribute][row])
                    count++;
            }

            //Random function is adding some randomness
            Assert.IsTrue(count >= 8 && count <= 9); 
        }



        [TestMethod]
        public void SVM_kernel_rbf_smo_all_training_samples()
        {
            initData_dataset_linear_subgd_jason_example();
            BuildSVMKernelSMO build = new BuildSVMKernelSMO();

            setPrivateVariablesInBuildObject(build);
            //Set params
            build.SetParameters(2, .0001, .001);
            build.setKernel(BuildSVMKernelSMO.Kernels.RadialBasisFunction, 1);
            ModelSVMKernelSMO model =
                (ModelSVMKernelSMO)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < _trainingData[0].Length; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (value == _trainingData[_indexTargetAttribute][row])
                    count++;
            }

            //Random function is adding some non-determinism to final values
            Assert.IsTrue(count >= 9 && count <= 10); //90-100% accuracy
        }
    }
}
