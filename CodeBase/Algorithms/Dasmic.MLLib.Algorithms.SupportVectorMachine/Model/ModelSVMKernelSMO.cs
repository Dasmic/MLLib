using Dasmic.Portable.Core;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    /// <summary>
    /// Solve Kernel based SVM using Sequential Minimal Optimization
    /// 
    /// Sequential Minimal Optimization is a state-of-the-art method for
    /// Quadritic Programming (QP) problems
    /// </summary>
    public class ModelSVMKernelSMO : ModelBase
    {
        public ModelKernel ModelKernel;


        public ModelSVMKernelSMO(double missingValue,
                             int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
            
        }

        public override
            double RunModelForSingleData(double[] data)
        {
            //data should not have target value
            double value = 0;int idx=0;
            foreach (double[] sv  in ModelKernel.SupportVectors)
            {
               
                value += ModelKernel.Alphas[idx] *
                    ModelKernel.TargetValues[idx] *
                    ModelKernel.Kernel.compute(data, sv);
                idx++;
            }
            value -= ModelKernel.Threshold;

            if (value > 0) return 1;
            return -1;           
        }

        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }
    }
}
