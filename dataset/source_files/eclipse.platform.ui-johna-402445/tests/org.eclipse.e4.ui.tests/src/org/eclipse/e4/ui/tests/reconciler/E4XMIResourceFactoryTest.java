/*******************************************************************************
 * Copyright (c) 2009, 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 ******************************************************************************/

package org.eclipse.e4.ui.tests.reconciler;

import java.util.Collection;
import org.eclipse.e4.ui.internal.workbench.ModelReconcilingService;
import org.eclipse.e4.ui.model.application.MApplication;
import org.eclipse.e4.ui.model.application.commands.MCommand;
import org.eclipse.e4.ui.model.application.commands.impl.CommandsFactoryImpl;
import org.eclipse.e4.ui.model.application.ui.basic.MPart;
import org.eclipse.e4.ui.model.application.ui.basic.MWindow;
import org.eclipse.e4.ui.model.application.ui.basic.impl.BasicFactoryImpl;
import org.eclipse.e4.ui.workbench.modeling.IModelReconcilingService;
import org.eclipse.e4.ui.workbench.modeling.ModelDelta;
import org.eclipse.e4.ui.workbench.modeling.ModelReconciler;

public class E4XMIResourceFactoryTest extends ModelReconcilerTest {

	public void testNonConflictingIds() {
		MApplication application = createApplication();

		saveModel();

		application = createApplication();

		MWindow window = BasicFactoryImpl.eINSTANCE.createWindow();
		application.getChildren().add(window);

		assertFalse(getId(application).equals(getId(window)));
	}

	public void testNonConflictingIds2() {
		MApplication application = createApplication();

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		MWindow window1 = BasicFactoryImpl.eINSTANCE.createWindow();
		application.getChildren().add(window1);

		Object state = reconciler.serialize();

		application = createApplication();

		MWindow window2 = BasicFactoryImpl.eINSTANCE.createWindow();
		application.getChildren().add(window2);

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(1, application.getChildren().size());
		assertEquals(window2, application.getChildren().get(0));

		applyAll(deltas);

		window1 = application.getChildren().get(0);
		window2 = application.getChildren().get(1);

		String applicationId = getId(application);
		String window1Id = getId(window1);
		String window2Id = getId(window2);

		assertFalse(applicationId.equals(window1Id));
		assertFalse(applicationId.equals(window2Id));
		assertFalse(window1Id.equals(window2Id));
	}

	public void testNonConflictingIds3_Bug303841() {
		MApplication application = createApplication();

		saveModel();

		ModelReconciler reconciler = createModelReconciler();
		reconciler.recordChanges(application);

		MCommand command = CommandsFactoryImpl.eINSTANCE.createCommand();
		command.setElementId("id");
		application.getCommands().add(command);

		MWindow window = BasicFactoryImpl.eINSTANCE.createWindow();
		application.getChildren().add(window);

		MPart part = BasicFactoryImpl.eINSTANCE.createPart();
		part.setElementId("id");
		window.getChildren().add(part);
		window.setSelectedElement(part);

		Object state = reconciler.serialize();

		application = createApplication();

		Collection<ModelDelta> deltas = constructDeltas(application, state);

		assertEquals(0, application.getChildren().size());
		assertEquals(0, application.getCommands().size());

		applyAll(deltas);
	}

	@Override
	protected IModelReconcilingService getModelReconcilingService() {
		return new ModelReconcilingService();
	}

}
