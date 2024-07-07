#ifndef __DATALINK_R_H
#define __DATALINK_R_H

template <typename T>
class DataLinkResult
{
public:
    std::unique_ptr<T[]> data;
    long size;
    bool valid;
};

#endif