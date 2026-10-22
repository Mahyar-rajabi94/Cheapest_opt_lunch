final_mean = np.asarray(es.result.xfavorite)
final_draws = es.ask(number=8)
final_mean_value = evaluate(final_mean)
final_values = [evaluate(point) for point in final_draws]
print(json.dumps({'mean_value': final_mean_value, 'draw_best': min(final_values),
                  'draw_median': float(np.median(final_values)), 'draw_worst': max(final_values),
                  'complete_population_update': False}), flush=True)
