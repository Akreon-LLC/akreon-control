add_executable(clf_cbf_qp_acc main.cpp)
target_link_libraries(clf_cbf_qp_acc PRIVATE Akreon::Control)
add_test(NAME clf_cbf_qp_acc COMMAND clf_cbf_qp_acc)
