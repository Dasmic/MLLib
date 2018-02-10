using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class NeuralNetworkConfigurationNotReady : Exception
    {
        public NeuralNetworkConfigurationNotReady(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public NeuralNetworkConfigurationNotReady()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_neural_net_configuration;
            }
        }
    }
}
