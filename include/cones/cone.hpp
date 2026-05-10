#ifndef CONE_CONES_H
#define CONE_CONES_H

/// @brief Base class for all cones.
class Cone{
protected:
    int barrier_parameter, num_params;
    bool is_complex, is_symmetric;
};

#endif