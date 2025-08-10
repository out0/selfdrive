#include <gtest/gtest.h>
#include "../../include/coord_conversion.h"
#include "test_utils.h"
#include <cmath>

#define TST_COORD_WIDTH 256
#define TST_COORD_HEIGHT 256
#define TST_COORD_REAL_W 34.641016151377535
#define TST_COORD_REAL_H 34.641016151377535

TEST(CoordConversionWorldPose, TestWorld_MapConversion)
{
    WorldPose origin(angle::rad(-4.256008878655848e-09),
                     angle::rad(-1.5864868596990013e-08),
                     1.0150023698806763,
                     angle::deg(89.999237523291130));

    CoordinateConverter conv(origin,
                             TST_COORD_WIDTH,
                             TST_COORD_HEIGHT,
                             TST_COORD_REAL_W,
                             TST_COORD_REAL_H);

    WorldPose p0(angle::deg(6.008207265040255e-10),
                 angle::deg(-9.445475428916048e-09),
                 1.0149040222167969,
                 angle::deg(90.01170944871556));

    MapPose rp0(-0.0010514655150473118,
                -6.688445864710957e-05,
                1.0149040222167969,
                angle::deg(0.011707369238138199));

    ASSERT_TRUE(conv.convert(p0) == rp0);

    WorldPose p1(angle::deg(-8.14085865386005e-09),
                 angle::deg(-1.782302467980099e-08),
                 1.0143463611602783,
                 angle::deg(359.9802297900763));
                 
    MapPose rp1(-0.0019840500317513943,
                0.0009062352473847568,
                1.0143463611602783,
                angle::deg(-90.00768280029297));

    MapPose pp = conv.convert(p1);

    ASSERT_TRUE(conv.convert(p1) == rp1);

    WorldPose p2(angle::deg(-1.3355261785363837e-08),
                 angle::deg(-2.011051673862556e-08),
                 1.013861060142517,
                 angle::deg(314.99925402977027));

    MapPose rp2(-0.002238692482933402,
                0.0014867002610117197,
                1.013861060142517,
                angle::deg(-135.0007781982422));

    ASSERT_TRUE(conv.convert(p2) == rp2);

    WorldPose p3(angle::deg(3.5947697853089267e-05),
                 angle::deg(-3.592238022246919e-05),
                 1.0108108520507812,
                 angle::deg(315.0036526715958));
    MapPose rp3(-3.998861074447632,
                -4.001679420471191,
                1.0108108520507812,
                angle::deg(-134.99636840820312));

    ASSERT_TRUE(conv.convert(p3) == rp3);

    WorldPose p4(angle::deg(0.00017966245712841555),
                 angle::deg(-3.5935588398026246e-05),
                 1.0016272068023682,
                 angle::deg(0.0));
    MapPose rp4(-4.000331401824951,
                -19.99993324279785,
                1.0016272068023682,
                angle::deg(-89.99999237060547));

    ASSERT_TRUE(conv.convert(p4) == rp4);

    WorldPose p5(angle::deg(-0.00017966285120962766),
                 angle::deg(-3.593243145773581e-05),
                 1.0016334056854248,
                 angle::deg(180.00000500895632));
    MapPose rp5(-3.9999799728393555,
                19.999977111816406,
                1.0016334056854248,
                angle::deg(90.00001525878906));

     ASSERT_TRUE(conv.convert(p5) == rp5);

    WorldPose p6(angle::deg(-0.0005389866003611132),
                 angle::deg(4.491557144842781e-05),
                 1.0365043878555298,
                 angle::deg(90.00011178750489));

    MapPose rp6(4.999978542327881,
                59.99971389770508,
                1.0365043878555298,
                angle::deg(0.00010962043597828597));

    ASSERT_TRUE(conv.convert(p6) == rp6);

    WorldPose p7(angle::deg(-0.0005389890676639197),
                 angle::deg(4.4915811324487864e-05),
                 1.036553144454956,
                 angle::deg(180.01979889717103));
    MapPose rp7(5.00000524520874,
                59.9999885559082,
                1.036553144454956,
                angle::deg(90.00000762939453));

    ASSERT_TRUE(conv.convert(p7) == rp7);

    WorldPose p8(angle::deg(-0.000538989855826344),
                 angle::deg(4.491587986050503e-05),
                 1.0365053415298462,
                 angle::deg(225.00000967629));
    MapPose rp8(5.0000128746032715,
                60.00007629394531,
                1.0365053415298462,
                angle::deg(135.0));

    ASSERT_TRUE(conv.convert(p8) == rp8);
}


TEST(CoordConversionWorldPose, TestMap_WorldConversion)
{
    WorldPose origin(angle::rad(-4.256008878655848e-09),
                     angle::rad(-1.5864868596990013e-08),
                     1.0150023698806763,
                     angle::deg(89.999237523291130));

    CoordinateConverter conv(origin,
                             TST_COORD_WIDTH,
                             TST_COORD_HEIGHT,
                             TST_COORD_REAL_W,
                             TST_COORD_REAL_H);

    WorldPose rp0(angle::deg(6.008207265040255e-10),
                 angle::deg(-9.445475428916048e-09),
                 1.0149040222167969,
                 angle::deg(90.01170944871556));

    MapPose p0(-0.0010514655150473118,
                -6.688445864710957e-05,
                1.0149040222167969,
                angle::deg(0.011707369238138199));

    ASSERT_TRUE(conv.convert(p0) == rp0);

    WorldPose rp1(angle::deg(-8.14085865386005e-09),
                 angle::deg(-1.782302467980099e-08),
                 1.0143463611602783,
                 angle::deg(359.9802297900763));
                 
    MapPose p1(-0.0019840500317513943,
                0.0009062352473847568,
                1.0143463611602783,
                angle::deg(-90.00768280029297));

    ASSERT_TRUE(conv.convert(p1) == rp1);

    WorldPose rp2(angle::deg(-1.3355261785363837e-08),
                 angle::deg(-2.011051673862556e-08),
                 1.013861060142517,
                 angle::deg(314.99925402977027));

    MapPose p2(-0.002238692482933402,
                0.0014867002610117197,
                1.013861060142517,
                angle::deg(-135.0007781982422));

    ASSERT_TRUE(conv.convert(p2) == rp2);

    WorldPose rp3(angle::deg(3.5947697853089267e-05),
                 angle::deg(-3.592238022246919e-05),
                 1.0108108520507812,
                 angle::deg(315.0036526715958));
    MapPose p3(-3.998861074447632,
                -4.001679420471191,
                1.0108108520507812,
                angle::deg(-134.99636840820312));

    ASSERT_TRUE(conv.convert(p3) == rp3);

    WorldPose rp4(angle::deg(0.00017966245712841555),
                 angle::deg(-3.5935588398026246e-05),
                 1.0016272068023682,
                 angle::deg(0.0));
    MapPose p4(-4.000331401824951,
                -19.99993324279785,
                1.0016272068023682,
                angle::deg(-89.99999237060547));

    ASSERT_TRUE(conv.convert(p4) == rp4);

    WorldPose rp5(angle::deg(-0.00017966285120962766),
                 angle::deg(-3.593243145773581e-05),
                 1.0016334056854248,
                 angle::deg(180.00000500895632));
    MapPose p5(-3.9999799728393555,
                19.999977111816406,
                1.0016334056854248,
                angle::deg(90.00001525878906));

     ASSERT_TRUE(conv.convert(p5) == rp5);

    WorldPose rp6(angle::deg(-0.0005389866003611132),
                 angle::deg(4.491557144842781e-05),
                 1.0365043878555298,
                 angle::deg(90.00011178750489));

    MapPose p6(4.999978542327881,
                59.99971389770508,
                1.0365043878555298,
                angle::deg(0.00010962043597828597));

    ASSERT_TRUE(conv.convert(p6) == rp6);

    WorldPose rp7(angle::deg(-0.0005389890676639197),
                 angle::deg(4.4915811324487864e-05),
                 1.036553144454956,
                 angle::deg(180.01979889717103));
    MapPose p7(5.00000524520874,
                59.9999885559082,
                1.036553144454956,
                angle::deg(90.00000762939453));

    ASSERT_TRUE(conv.convert(p7) == rp7);

    WorldPose rp8(angle::deg(-0.000538989855826344),
                 angle::deg(4.491587986050503e-05),
                 1.0365053415298462,
                 angle::deg(225.00000967629));
    MapPose p8(5.0000128746032715,
                60.00007629394531,
                1.0365053415298462,
                angle::deg(135.0));

    ASSERT_TRUE(conv.convert(p8) == rp8);
}
