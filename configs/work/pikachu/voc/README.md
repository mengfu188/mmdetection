
## v1.1

- configs/work/pikachu/retinanet_r50_fpn_2_2.py

- configs/work/pikachu/retinanet_r50_fpn_2_3.py

- model.bbox_head.num_classes != len(dataset.CLASSES)

```
/home/cmf/anaconda3/bin/python /home/cmf/w_public/mmdetection.cmf/tools/test.py configs/work/pikachu/retinanet_r50_fpn_2_3.py work_dirs/work/pikachu/retinanet_r50_fpn_2_3/latest.pth --eval mAP
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 100/100, 40.7 task/s, elapsed: 2s, ETA:     0sTraceback (most recent call last):
  File "/home/cmf/w_public/mmdetection.cmf/tools/test.py", line 175, in <module>
    main()
  File "/home/cmf/w_public/mmdetection.cmf/tools/test.py", line 171, in main
    dataset.evaluate(outputs, args.eval, **kwargs)
  File "/home/cmf/w_public/mmdetection.cmf/mmdet/datasets/custom.py", line 197, in evaluate
    logger=logger)
  File "/home/cmf/w_public/mmdetection.cmf/mmdet/core/evaluation/mean_ap.py", line 385, in eval_map
    mean_ap, eval_results, dataset, area_ranges, logger=logger)
  File "/home/cmf/w_public/mmdetection.cmf/mmdet/core/evaluation/mean_ap.py", line 448, in print_map_summary
    label_names[j], num_gts[i, j], results[j]['num_dets'],
IndexError: tuple index out of range
```