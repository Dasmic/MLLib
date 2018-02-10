using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.InstanceBased
{
    public class BuildSelfOrganizingMap:BuildBase
    {
        protected long _xdim;
        protected long _ydim;
        protected long _distance;
        protected double _origRadius;
        protected int _useDistanceInfluence;

        public BuildSelfOrganizingMap()
        {
            _xdim = 2;
            _ydim = 2;
            _weightBaseValue = .001;
            _maxEpoch = 1000;
            _origRadius = double.MaxValue;
            _useDistanceInfluence = 0;
        }

        /// <summary>
        /// <para>0 - xdim;default=2</para>
        /// <para>1 - ydim;default=2</para>
        /// <para>2 - Learning Rate; default=.5</para>
        /// <para>3 - Max Epoch;default=1000</para>
        /// <para>4 - Use distance influence in Weight Updates;0=false,1=true;default=0</para>
        /// <para>5 - Original Radius</para>
        /// <para>6 - Weight Base Value, used in initialization;default=.001</para>      
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            int idx = 0;
            if (values.Length > idx)
                if (values[idx] != double.NaN)
                    _xdim = (long)values[0];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _ydim = (long)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _learningRate = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)                    
                    _maxEpoch = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _useDistanceInfluence = (int)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _origRadius = (double)values[idx];
            if (values.Length > ++idx)
                if (values[idx] != double.NaN)
                    _weightBaseValue = (double)values[idx];
                        
        }


        /// <summary>
        /// Builds a Self-Organizing Map
        /// The Model contains the weight vectors of the SOM
        /// </summary>
        /// <param name="trainingData"></param>
        /// <param name="attributeHeaders"></param>
        /// <param name="indexTargetAttribute">Not required for SOM</param>
        /// <returns></returns>
        public override Common.MLCore.ModelBase
                BuildModel(double[][] trainingData,
                string[] attributeHeaders,
                int indexTargetAttribute=-1)
        {
            if (indexTargetAttribute == -1)
                indexTargetAttribute = attributeHeaders.Length - 1;
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelSelfOrganizingMap model = new ModelSelfOrganizingMap(_missingValue,
                                                indexTargetAttribute,
                                                trainingData.Length);
            model.SomMap = new SingleSOMNode[_xdim][];
            long noOfWeights = trainingData.Length; //Equal to number of features

            //Choose as lower of dimension as distance diamter
            long distanceDiameter = (_xdim < _ydim) ? _xdim : _ydim;
            double origRadius;

            if (_origRadius == double.MaxValue) //Not set by user
            {
                origRadius = distanceDiameter / 2.0; //Radius of influence
                //Make it circluar so that Maps in immediate grid can be accomodated
                //origRadius = sqrt(x*x+y*y), x=y=oriRadiud 
                origRadius = origRadius * Math.Sqrt(2.0);
            }
            else
                origRadius = _origRadius;

            double[][] computedDistValues = new double[_xdim][];

            //Initialize SOM Weights
            InitializeSOMWeights(model,noOfWeights);

            double learningRate = _learningRate;
            double radius = origRadius;
            double timeConstant = (double)_maxEpoch / Math.Log(origRadius);
            //Start Convergence Loops            
            for (int epoch = 0; epoch < _maxEpoch; epoch++) //Do not parallelize
            {                
                for (int dataRow = 0; dataRow < trainingData[0].Length; dataRow++)
                {
                    double[] lowestDistance = new double[_xdim];
                    long[] lowestDistanceY = new long[_xdim];
                    double[] inputDataRow = GetLinearArray(_trainingData, dataRow, _trainingData.Length - 1);
                    //Find Best Matching Unit (BMU) for training row
                    //Find lowest distance in xdim
                    Parallel.For(0, _xdim,
                        new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, xdim =>
                        //for (long xdim = 0; xdim < _xdim; xdim++) //Parallelize this loop
                    {
                        double computedDistValue = double.MaxValue;
                        lowestDistance[xdim] = double.MaxValue;
                        for (long ydim = 0; ydim < _ydim; ydim++)
                        {
                            //computedDistValues[xdim][ydim] =
                            computedDistValue =
                               model.SomMap[xdim][ydim].GetComputedDistance(inputDataRow);
                            if (computedDistValue < lowestDistance[xdim])
                            {
                                lowestDistance[xdim] = computedDistValue;
                                lowestDistanceY[xdim] = ydim;
                            }
                        }//ydim
                    });//xdim

                    long bmuX=0, bmuY=0;
                    double finalDistanceValue = double.MaxValue;
                    for (int xdim = 0; xdim < _xdim; xdim++) //Parallelize this loop
                    {
                        if(lowestDistance[xdim] < finalDistanceValue)
                        {
                            bmuX = xdim;
                            bmuY = lowestDistanceY[xdim];
                            finalDistanceValue = lowestDistance[xdim];
                        }
                    }

                  
                    //Update Weights in neighborhood accounting for distance from BMU
                    for (long xdim = 0; xdim < _xdim; xdim++) //Parallelize this loop
                    {
                        for (long ydim = 0; ydim < _ydim; ydim++)
                        {
                            model.SomMap[xdim][ydim].UpdateWeights(model.SomMap[bmuX][bmuY],
                                                    inputDataRow, radius, learningRate,_useDistanceInfluence==1?true:false);
                                                                    
                        }
                    }
                } //row

                //Update learning rate and distance
                //Formula from: http://www.ai-junkie.com/ann/som/som3.html                
                radius = origRadius * Math.Exp(-(double)epoch / timeConstant); // Math.Pow(origRadius,((double)epoch/(double)_maxEpoch)); //;                    
                learningRate = _learningRate * Math.Exp(-(double)epoch /timeConstant);                
            }

            return model;
        } //buildModel


        protected void InitializeSOMWeights(ModelSelfOrganizingMap model, 
                                                long noOfWeights)
        {
            Parallel.For(0, _xdim,
                            new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                            xdim =>
                            {
                                Random rnd = new Random();
                                model.SomMap[xdim] = new SingleSOMNode[_ydim];
                                //computedDistValues[xdim] = new double[_ydim];
                                for (int ydim = 0; ydim < _ydim; ydim++)
                                {
                                    model.SomMap[xdim][ydim] = new SingleSOMNode(noOfWeights, xdim, ydim);
                                    model.SomMap[xdim][ydim].InitWeights(rnd, _weightBaseValue);
                                }
                            });
        }
    }
}
