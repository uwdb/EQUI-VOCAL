from vocal.vocal import Vocal
from vocal.train_and_eval_proxy_model import TrainAndEvalProxyModel, ClevrerEdge


if __name__ == '__main__':
    # logging.basicConfig(filename="output.log", encoding="utf-8", level=logging.INFO)
    # cevdb = TrainAndEvalProxyModel(dataset="clevrer", query="clevrer_near", temporal_heuristic=False, budget=1000, frame_selection_method="random", thresh=1.0)
    cevdb = TrainAndEvalProxyModel(dataset="clevrer", query="clevrer_far", temporal_heuristic=False, budget=1000, frame_selection_method="random", thresh=2.0)
    cevdb = ClevrerEdge(dataset="clevrer", query="clevrer_edge", temporal_heuristic=False, budget=1000, frame_selection_method="least_confidence", thresh=20)
    # cevdb = ComplexEventVideoDB(dataset="clevrer", query="clevrer_collision", temporal_heuristic=False)
    plot_data_y_annotated, plot_data_y_materialized = cevdb.run()
    cevdb.save_data(plot_data_y_annotated)
    print("plot_data_y_annotated", plot_data_y_annotated)
    print("plot_data_y_materialized", plot_data_y_materialized)
    # plot_data_y_annotated_list = []
    # plot_data_y_materialized_list = []
    # for i in range(20):
    #     print("Iteration: ", i)
    #     # cevdb = ComplexEventVideoDB(dataset="meva", query="meva_person_enters_vehicle", temporal_heuristic=False)
    #     cevdb = ComplexEventVideoDB(dataset="visualroad_traffic2", query="turning_car_and_pedestrain_at_intersection", temporal_heuristic=True)
    #     # cevdb.tsne_plot()
    #     # cevdb = FilteredProcessing(dataset="meva", query="meva_person_embraces_person", temporal_heuristic=True)
    #     plot_data_y_annotated, plot_data_y_materialized = cevdb.run()
    #     plot_data_y_annotated_list.append(plot_data_y_annotated)
    # cevdb.save_data(plot_data_y_annotated_list)
