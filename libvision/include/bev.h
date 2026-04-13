#include <driveless/search_frame.h>
#include <stdexcept>

class BEV {
    int _width;
    int _height;
    std::pair<int, int> _carSizePx;
    SearchFrame *_data;
    int _carClassCode;

public:

    BEV(int width, int height, std::pair<int, int> carSizePx, int carClassCode);

    void compute (
        SearchFrame *front,
        SearchFrame *back,
        SearchFrame *left,
        SearchFrame *right
    );

    inline SearchFrame *get() {
        return _data;
    }

};

