ents.items():
                client.test_acc[-1] /= num_test_iterations
                overall_acc[-1] += client.test_acc[-1]
            
            overall_acc[-1] /= num_clients
            # print(f'Acc for epoch {epoch+1}: {overall_acc[-1]}')
            print(f'Test Acc: {overall_acc[-1]}')