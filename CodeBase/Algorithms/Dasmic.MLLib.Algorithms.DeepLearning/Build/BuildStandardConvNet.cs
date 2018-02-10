using System.Threading.Tasks;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    /// <summary>
    /// Generic Structure: INPUT - CONV - RELU - POOL - FC
    /// </summary>
    public class BuildConvNet:BuildConvNetBase
    {
        //Convolution
        public int NoOfFeatures { get; set; }
        public int SizeOfFeatures { get; set; }
        public int StrideConvolution  { get; set; }

        //Default is ReLU
        protected IActivationFunction ActivationFunctionConvolution { get; set; }

        //Pooling
        private int _sizeWindow;
        private int _strideWindow;

        //noOfUnits 

        public BuildConvNet()
        {
            ActivationFunctionConvolution = new RectifiedLinearUnit();
        }

        //Initialize modle by passing target values of training data
        public override void InitializeModel(string[] targetHeaders)
        {

        }
                       


        /// <summary>
        /// 
        /// </summary>
        /// <param name="trainingData">3D array having a single training data with depth in 3rd Dimension. 
        /// Indexed as [depth][columns][rows]</param> 
        /// 
        /// This function will have to called for each training data
        /// 
        /// Call InitModel3D before calling this
        /// 
        /// This different approach has been adopted since training data could be very huge to be accomodated in memory
        /// This way only one training data needs to be present in memory at any given time
        /// <param name="indexTargetAttribute"></param>
        /// <returns></returns>
        public override Common.MLCore.ModelBase
           BuildModelSingle(double[][][] trainingData,
                        Dictionary<double, string> targetValueMapping)
        {
            //call for all training values

            return null;
        }

        protected bool VerifyUpstreamLayers(ModelConvNetBase model)
        {
            bool exceptionFlag = false;
            Parallel.For(1, model.GetNumberOfLayers(),
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (model.GetLayer(idxLayer).GetNumberOfUpstreamUnits() < 0)//Check if Upstream Layer is setup, this will set an exception otherwise
                           exceptionFlag = true;
                   });
            return exceptionFlag;
        }

        protected bool VerifyLayers(ModelBackPropagationBase model)
        {
            bool exceptionFlag = false;
            Parallel.For(0, model.GetNumberOfLayers() - 1,
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (model.GetLayer(idxLayer) == null)
                           exceptionFlag = true;
                   });
            return exceptionFlag;
        }

    }
}
