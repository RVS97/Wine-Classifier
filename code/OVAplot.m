%% one vs all plot
% simple plot for demonstration of one-vs-all principle
 points = [2.1, 2.3, 1.6, 2, 2.2, 1.9, 1.1, 0.9, 1.1, 0.5, 1.3, 1, 2.4, 1.6, 2, 2.2, 1.9, 2.3];
 set1 = linspace(1, 3, 6);
 set2 = linspace(4, 6, 6);
 set3 = linspace(7, 9, 6); 
 figure 
 plot(set1,points(1:6), 'xb')
 hold on
 plot(set2,points(7:12), 'og')
 plot(set3,points(13:18), '^k')
 axis([0 10 0 3])
 set(gca, 'Fontsize', 18)
 title('OVA initial classes', 'Fontsize', 25)
 
 figure
 plot(set1,points(1:6), 'xr')
 hold on
 plot(set2,points(7:12), 'og')
 plot(set3,points(13:18), '^r')
 delimiter = 1.5*ones(1,9);
 plot(1:9, delimiter, '--r')
 axis([0 10 0 3])
 set(gca, 'Fontsize', 18)
 title('OVA training', 'Fontsize', 25)
 
 %% poslin plot for neural network transfer function
 points = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5];
 ax = -5:5;
 figure
 plot(ax, points, 'Linewidth', 2);
 axis([-5 5 -1 5])
 grid on
 xlabel('x')
 ylabel('y')
 set(gca, 'Fontsize', 25)
 title('y = poslin(x)', 'Fontsize', 45)