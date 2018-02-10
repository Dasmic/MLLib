using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.SupportVectorMachine
{
    public class BuildSVMKernelSMO:BuildBase
    {       
        private double _C;
        private double _tolerance;
        private double _eps;
        private IKernel _kernel;

        public enum Kernels {Linear, Polynomial, RadialBasisFunction}

        public BuildSVMKernelSMO() : base()
        {
            //_maxIterations = 10000; Not used
            _C = 2;
            _tolerance = .001;
            _eps = .001;
            setKernel(Kernels.Linear, 0); //Set default Kernel as linear
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="kernelType"></param>
        /// <param name="parameter">Polynomial:degree, RBF:gamma</param>
        public void setKernel(Kernels kernelType, double parameter)
        {
            switch(kernelType)
            {
                case Kernels.Linear:
                    _kernel = new KernelLinear();
                    break;
                case Kernels.Polynomial:
                    _kernel = new KernelPolynomial((int)parameter);
                    break;
                case Kernels.RadialBasisFunction:
                    _kernel = new KernelRadialBasisFunction(parameter);
                    break;
                default:
                    _kernel = new KernelLinear();
                    break;                   
            }

        }

        /// <summary>
        /// Sets a custom Kernel 
        /// Custom Kernel should implement IKernel
        /// </summary>
        /// <param name="kernel"></param>
        public void setCustomKernel(IKernel kernel)
        {
            _kernel = kernel;
        }


        /// <summary>
        /// 0:C, default = 2
        /// 1:tolerance, default = .0001
        /// 2:eps, default = .001
        /// </summary>
        /// <param name="values"></param>
        public override void
             SetParameters(params double[] values)
        {
            if (values.Length > 0)
                if (values[0] != double.NaN) _C = values[0];
            if (values.Length > 1)
                if (values[1] != double.NaN) _tolerance = values[1];
            if (values.Length > 2)
                if (values[2] != double.NaN) _eps = values[2];
            //if (values.Length > 3)
             //   if (values[3] != double.NaN) _maxIterations = (long) values[3];
        }

        public override Common.MLCore.ModelBase BuildModel(
                             double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelSVMKernelSMO model =
                            new ModelSVMKernelSMO(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            SequentialMinimalOptimization smo =
                new SequentialMinimalOptimization();

            smo.setParameters(_C, _tolerance, _eps, _maxIterations);
            model.ModelKernel = smo.computeSupportVectors(trainingData, 
                                            indexTargetAttribute,
                                            _kernel, 
                                            _maxParallelThreads);

            return model;
        }

    }
}
