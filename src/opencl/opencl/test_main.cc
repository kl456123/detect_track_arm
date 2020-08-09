#include <gtest/gtest.h>
#include <glog/logging.h>



GTEST_API_ int main(int argc, char** argv) {
    // disable log during test
    // FLAGS_logtostderr = true;
    // FLAGS_minloglevel = 5;
    google::InitGoogleLogging(argv[0]);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
