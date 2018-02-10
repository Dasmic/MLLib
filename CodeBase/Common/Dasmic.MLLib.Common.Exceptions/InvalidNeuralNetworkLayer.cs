using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidNeuralNetworkLayer : Exception
    {
        public InvalidNeuralNetworkLayer(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public InvalidNeuralNetworkLayer()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_neural_net_layer;
            }
        }
    }
}
