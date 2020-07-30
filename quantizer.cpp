

extern C
{
#define radius 32768
template<class T>
int quantize_api(T &data, T pred, double error_bound) {

    T diff = data - pred;
    int quant_index = (int) (fabs(diff) / error_bound) + 1;
    if (quant_index < radius * 2) {
        quant_index >>= 1;
        int half_index = quant_index;
        quant_index <<= 1;
        int quant_index_shifted;
        if (diff < 0) {
            quant_index = -quant_index;
            quant_index_shifted = radius - half_index;
        } else {
            quant_index_shifted = radius + half_index;
        }
        T decompressed_data = pred + quant_index * error_bound;
        if (fabs(decompressed_data - data) > error_bound) {
            return 0;
        } else {
            data = decompressed_data;
            return quant_index_shifted;
        }
    } else {
        return 0;
    }
}

int quantize_api_int(int &data,int pred,double error_bound){
    return quantize_api<int>(data,pred,error_bound);
}
int quantize_api_float(float &data,float pred,double error_bound){
    return quantize_api<float>(data,pred,error_bound);
}
int quantize_api_double(double &data,double pred,double error_bound){
    return quantize_api<double>(data,pred,error_bound);
}

}

