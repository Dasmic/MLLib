using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// This is a 2-layer Neural Network that uses Sigmod function at both hidden and output
    /// layer
    /// </summary>
    public class Build2LBackPropagation : BuildBackPropagationBase
    {
        int _noOfUnitsHiddenLayer;
        int _scalingFactor; //For setting number of hidden nodes       

        public double[][] WeightsHidden { get; set; }

        public Build2LBackPropagation()
        {
            _mode = ModelBackPropagationBase.EnumMode.Regression;
            _noOfUnitsHiddenLayer = int.MinValue;
            _scalingFactor = 2;
            _activationFunctions = new IActivationFunction[2];
        }

        /// <summary>
        /// <para>0 - Mode;default=0: 0 - Regression (one output unit), 1 - Classification (multiple output neurons)</para>
        /// <para>1 - Threshold;default=.01</para>
        /// <para>2 - Maximum Epoch/Iterations; default=1000</para>
        /// <para>3 - Alpha (Initial Learning Rate);default=.3</para>
        /// <para>4 - Scaling factor for automatically setting number of hidden units;default=2</para>
        /// <para>5 - Number of units in Hidden Layer</para>
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            int idx = 0;
            if (values.Length > idx)
                if (values[idx] != double.NaN)
                    _mode = (ModelBackPropagationBase.EnumMode)(int)values[0];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _threshold = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _noOfEpoch = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _alpha = values[idx];            
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _scalingFactor = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _noOfUnitsHiddenLayer = (int)values[idx];
        }

        /// <summary>
        /// Manually set Activation Functions, Index:
        /// 0 - Hidden Layer
        /// 1 - Custom Layer
        /// </summary>
        /// <param name="idxLayer"></param>
        /// <param name="activationFunction"></param>
        public void SetActivationFunction(int idxLayer,
                        IActivationFunction activationFunction)
        {
            if (idxLayer > _activationFunctions.Length ||
                idxLayer < 0)
                throw new IndexOutOfRangeException();

            _activationFunctions[idxLayer] = activationFunction;            
        }
       
        public override Common.MLCore.ModelBase
            BuildModel(double[][] trainingData,
                         string[] attributeHeaders,
                         int indexTargetAttribute)
        {            
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            //model.Mode = _mode; //Classification or Regression
            //Find out number of categories            
            _noOfUnitsOutputLayer = GetNumberOfTargetValues(_mode, trainingData, indexTargetAttribute).Length;


            //Set Number of Hidden Layers
            if (_noOfUnitsHiddenLayer == int.MinValue)
            {
                _noOfUnitsHiddenLayer=
                    GetNumberOfHiddenUnits(_noOfUnitsOutputLayer, _scalingFactor,_noOfAttributes);
            }

            if (_activationFunctions[0] == null)
                SetActivationFunction(0, new Sigmoid());
            if (_activationFunctions[1] == null)
                SetActivationFunction(1, new Sigmoid());

            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters((double)_mode,1,_threshold,_noOfEpoch,_alpha);            
            build.AddHiddenLayer(0, _noOfUnitsHiddenLayer, _activationFunctions[0]);
            build.SetOutputLayerActivationFunction(_activationFunctions[1]);

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);        
            return model;
        }       

    }//Function
}
